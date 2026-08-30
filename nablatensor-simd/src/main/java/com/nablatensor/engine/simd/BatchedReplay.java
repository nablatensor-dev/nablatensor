/*
 * Copyright 2026 The NablaTensor Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nablatensor.engine.simd;

import com.nablatensor.engine.AadOp;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.AbstractAadExecutable;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Tape flattening, batch splitting and worker threads shared by the two typed
 * sweeps.
 *
 * <p>Scenarios are the parallel axis at every level: SIMD lanes within a
 * vector, {@link SimdSupport#BATCH} scenarios within a sweep, and a contiguous
 * path range per worker thread. Splitting on scenarios rather than on the tape
 * is forced by the tape being a chain of dependent scalar operations, and it is
 * also what keeps the workers independent — each owns its own value and adjoint
 * arrays and shares nothing until the final sum.
 */
abstract class BatchedReplay extends AbstractAadExecutable {

  static final int BATCH = SimdSupport.BATCH;

  // Flattened once so the sweeps read plain arrays rather than calling through
  // the tape for every node of every batch.
  final AadOp[] ops;
  final int[] rowA;
  final int[] rowB;
  final int[] argA;
  final double[] constants;
  final boolean[] active;
  final int[] inputRow;
  final int outRow;

  private final int threads;
  private final ExecutorService pool;

  /**
   * {@code -Dnablatensor.crn=on}: common-random-numbers caching. The first
   * {@link #replay} for a given {@code (seed, pathOffset, paths)} generates the
   * whole draw block once and keeps it; a later replay of the same block — a
   * re-price under a shocked market — reuses it and skips the RNG. Bounded by
   * {@code -Dnablatensor.crn.cap} elements (default 1<<27 ≈ 1 GiB).
   */
  private static final boolean CRN = "on".equals(System.getProperty("nablatensor.crn"));
  private static final int CRN_CAP = Integer.getInteger("nablatensor.crn.cap", 1 << 27);

  final int randCount;
  private double[] drawCache;
  private long cacheSeed = Long.MIN_VALUE;
  private long cachePathOffset = Long.MIN_VALUE;
  private long cachePathsHeld = -1;
  private int cachePadded;
  private boolean cacheFilled;

  /** Per-{@link #runRange} draw source: a shared cache slice, or fresh generation. */
  static final class Draws {
    final double[] cache;
    final long origin;
    final int padded;
    final boolean write;

    Draws(double[] cache, long origin, int padded, boolean write) {
      this.cache = cache;
      this.origin = origin;
      this.padded = padded;
      this.write = write;
    }

    boolean read() {
      return cache != null && !write;
    }

    /** index of draw {@code k} for the batch starting at global path {@code base}. */
    int index(long base, int k) {
      return k * padded + (int) (base - origin);
    }

    static final Draws GENERATE = new Draws(null, 0, 0, false);
  }

  BatchedReplay(AadTape tape, AadOptions options) {
    super(tape, options);
    this.randCount = tape.randCount();
    this.threads = Math.max(1, options.resolvedThreads());
    this.pool = threads > 1
        ? Executors.newFixedThreadPool(threads, runnable -> {
            Thread thread = new Thread(runnable, "aad-simd");
            thread.setDaemon(true);
            return thread;
          })
        : null;

    int n = tape.size();
    this.ops = new AadOp[n];
    this.rowA = new int[n];
    this.rowB = new int[n];
    this.argA = new int[n];
    this.constants = new double[n];
    this.active = new boolean[n];
    for (int i = 0; i < n; i++) {
      ops[i] = tape.op(i);
      argA[i] = tape.argA(i);
      rowA[i] = tape.argA(i) * BATCH;
      rowB[i] = tape.argB(i) * BATCH;
      constants[i] = tape.constant(i);
      active[i] = tape.isActive(i);
    }
    this.inputRow = new int[tape.inputCount()];
    for (int j = 0; j < inputRow.length; j++) {
      inputRow[j] = tape.inputNode(j) * BATCH;
    }
    this.outRow = tape.outputNode() * BATCH;
  }

  /** Evaluates a contiguous path range on the calling thread. */
  abstract Accumulator runRange(long pathFrom, long count, long seed, Draws draws);

  @Override
  public final AadResult replay(long paths, long pathOffset, long seed) {
    checkOpen();
    if (paths <= 0) {
      throw new IllegalArgumentException("paths must be positive");
    }

    Draws draws = Draws.GENERATE;
    long padded = ((paths + BATCH - 1) / BATCH) * BATCH;
    if (CRN && randCount > 0 && padded * randCount <= CRN_CAP) {
      boolean hit = cacheFilled && seed == cacheSeed
          && pathOffset == cachePathOffset && paths == cachePathsHeld;
      if (hit) {
        draws = new Draws(drawCache, pathOffset, cachePadded, false);
      } else {
        int need = Math.toIntExact(padded * randCount);
        if (drawCache == null || drawCache.length < need) {
          drawCache = new double[need];
        }
        cacheSeed = seed;
        cachePathOffset = pathOffset;
        cachePathsHeld = paths;
        cachePadded = (int) padded;
        cacheFilled = false;
        draws = new Draws(drawCache, pathOffset, (int) padded, true);
      }
    }

    long start = System.nanoTime();
    final Draws d = draws;
    Accumulator total = threads == 1
        ? runRange(pathOffset, paths, seed, d)
        : runParallel(paths, pathOffset, seed, d);
    double seconds = (System.nanoTime() - start) / 1e9;

    if (draws.write) {
      cacheFilled = true;
    }

    double[] gradients = new double[tape.inputCount()];
    for (int j = 0; j < gradients.length; j++) {
      gradients[j] = total.gradient[j] / paths;
    }
    return new AadResult(total.value / paths, gradients, tape.inputNames(), paths, seconds);
  }

  private Accumulator runParallel(long paths, long pathOffset, long seed, Draws draws) {
    // Whole batches per worker, so only the last worker can have a partial one.
    long batches = (paths + BATCH - 1) / BATCH;
    long each = batches / threads;
    long extra = batches % threads;
    List<Callable<Accumulator>> tasks = new ArrayList<>(threads);
    long cursor = pathOffset;
    long remaining = paths;
    for (int t = 0; t < threads && remaining > 0; t++) {
      long count = Math.min((each + (t < extra ? 1 : 0)) * BATCH, remaining);
      long from = cursor;
      cursor += count;
      remaining -= count;
      if (count > 0) {
        tasks.add(() -> runRange(from, count, seed, draws));
      }
    }
    Accumulator total = new Accumulator(tape.inputCount());
    try {
      for (Future<Accumulator> future : pool.invokeAll(tasks)) {
        total.add(future.get());
      }
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("replay interrupted", interrupted);
    } catch (ExecutionException failure) {
      Throwable cause = failure.getCause();
      throw cause instanceof RuntimeException runtime ? runtime : new IllegalStateException(cause);
    }
    return total;
  }

  @Override
  public void close() {
    super.close();
    if (pool != null) {
      pool.shutdownNow();
    }
  }

  static final class Accumulator {
    double value;
    final double[] gradient;

    Accumulator(int inputs) {
      this.gradient = new double[inputs];
    }

    void add(Accumulator other) {
      value += other.value;
      for (int j = 0; j < gradient.length; j++) {
        gradient[j] += other.gradient[j];
      }
    }
  }
}
