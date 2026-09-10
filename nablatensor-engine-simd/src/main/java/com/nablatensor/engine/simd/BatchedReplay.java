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
import com.nablatensor.engine.AadTotals;
import com.nablatensor.engine.FlatTape;
import com.nablatensor.engine.HostAadExecutable;

/**
 * The batch layout and the common-random-numbers cache shared by the two typed
 * sweeps.
 *
 * <p>Scenarios are the parallel axis at every level: SIMD lanes within a
 * vector, {@link SimdSupport#BATCH} scenarios within a sweep, and a contiguous
 * path range per worker thread — the last of which is
 * {@link HostAadExecutable}'s job. Splitting on scenarios rather than on the
 * tape is forced by the tape being a chain of dependent scalar operations, and
 * it is also what keeps the workers independent: each owns its own value and
 * adjoint arrays and shares nothing until the final sum.
 *
 * <p>What this class adds over the shared base is the node-to-row remapping —
 * every sweep keeps {@link #BATCH} scenarios side by side per node, so it
 * indexes rows rather than nodes — and the draw cache.
 */
abstract class BatchedReplay extends HostAadExecutable {

  static final int BATCH = SimdSupport.BATCH;

  // The flat tape, plus the row indices the sweeps address it by.
  final AadOp[] ops;
  final int[] rowA;
  final int[] rowB;
  final int[] argA;
  final double[] constants;
  final boolean[] active;
  final int[] inputRow;
  final int outRow;

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
    super(tape, options, "aad-simd");
    this.randCount = tape.randCount();
    FlatTape flat = new FlatTape(tape);
    this.ops = flat.op;
    this.argA = flat.argA;
    this.constants = flat.constant;
    this.active = flat.active;
    this.rowA = FlatTape.scaled(flat.argA, BATCH);
    this.rowB = FlatTape.scaled(flat.argB, BATCH);
    this.inputRow = FlatTape.scaled(flat.inputNode, BATCH);
    this.outRow = flat.outputNode[0] * BATCH;
  }

  /** Evaluates a contiguous path range on the calling thread. */
  abstract AadTotals runRange(long pathFrom, long count, long seed, Draws draws);

  @Override
  public final AadResult replay(long paths, long pathOffset, long seed) {
    long start = beginReplay(paths);

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

    // Whole batches per worker, so only the last worker can get a partial one.
    final Draws d = draws;
    AadTotals total = runRanges(paths, pathOffset, BATCH,
        (from, count) -> runRange(from, count, seed, d));
    double seconds = (System.nanoTime() - start) / 1e9;

    if (draws.write) {
      cacheFilled = true;
    }
    return total.toResult(tape, paths, seconds);
  }
}
