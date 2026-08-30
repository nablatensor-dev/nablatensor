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
package com.nablatensor.engine.jit;

import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.AbstractAadExecutable;
import com.nablatensor.engine.JitOptimizations;
import com.nablatensor.engine.JitOptimizations.Category;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Replay backed by a bytecode kernel generated for this exact tape. One path is
 * one {@code forward} call (plus one {@code reverse} for Greeks) into
 * straight-line generated code — no interpreter dispatch. Scenarios are
 * independent, so the only parallelism is a contiguous path range per worker;
 * each worker owns its {@code v}/{@code d} scratch.
 *
 * <p>fp64 and fp32 share this class: the kernel arithmetic is {@code double} or
 * {@code float} per {@link AadOptions#precision()}, while the per-scenario value
 * and gradient totals are always accumulated in {@code double}.
 */
final class JitReplay extends AbstractAadExecutable {

  private final boolean f32;
  private final JitKernel kernel64;
  private final JitKernelF32 kernel32;
  private final int nodes;
  private final int vLen;
  private final int randCount;   // flat draw-buffer size = tape.randTotal()
  private final int[] streamNormal;
  private final int[] streamUniform;
  private final int[] streamOffset;
  private final int[] inputNode;   // remapped to v[]/d[] slots
  private final int[] outputNode;  // remapped, one per recorded output
  private final double compileSeconds;
  private final int classFileBytes;
  private final int scratchElems;
  private final int segNodes;

  /**
   * Whether the loop-rolling code generator ran for this tape. Enabled by
   * {@link Category#ROLLED_LOOPS} (via {@code .jit(...)}); the system property
   * {@code -Dnablatensor.jit.roll} overrides it either way ({@code on}/{@code off}).
   */
  private final boolean roll;

  /**
   * Whether the Box-Muller draw generator uses the polynomial approximations in
   * {@link JitFastMath}. Enabled by {@link Category#FAST_MATH}.
   */
  private final boolean fastMath;

  /**
   * Common-random-numbers caching. The first {@link #replay} for a given
   * {@code (seed, pathOffset, paths)} generates the whole draw block once and
   * keeps it; later replays of the same block — a re-price under a shocked
   * market — reuse it and skip the RNG. Enabled by {@link Category#DRAW_CACHE}
   * (via {@code .jit(...)}) or {@code -Dnablatensor.crn=on}. Bounded by
   * {@code -Dnablatensor.crn.cap} draw elements (default 1<<27); larger blocks
   * fall back to regenerating. In fp32 the cache is {@code float[]}, so it is
   * half the size.
   */
  private final boolean crn;

  private static final boolean SKIP_RNG = "zero".equals(System.getProperty("nablatensor.jit.randn"));
  private static final int CRN_CAP = Integer.getInteger("nablatensor.crn.cap", 1 << 27);

  private double[] cache64;
  private float[] cache32;
  private long cacheSeed = Long.MIN_VALUE;
  private long cachePathOffset = Long.MIN_VALUE;
  private long cachePaths = -1;
  private boolean cacheFilled;

  private final int threads;
  private final ExecutorService pool;

  JitReplay(AadTape tape, AadOptions options) {
    super(tape, options);
    this.f32 = options.precision() == AadOptions.Precision.FLOAT32;
    this.segNodes = Math.max(8, Integer.getInteger("nablatensor.jit.seg", 128));

    JitOptimizations jit = options.jit();
    String rollProp = System.getProperty("nablatensor.jit.roll");
    boolean rollWanted = "off".equals(rollProp) ? false
        : "on".equals(rollProp) ? true
        : jit.has(Category.ROLLED_LOOPS);
    // The loop-roller's per-iteration draw indexing assumes one stream of normals,
    // and its reverse assumes a single seeded output; uniforms, named streams and
    // multi-output tapes take the flat kernel, which handles any layout.
    this.roll = rollWanted && !tape.hasExtendedRandom() && tape.outputCount() == 1;
    this.fastMath = jit.has(Category.FAST_MATH);
    this.crn = "on".equals(System.getProperty("nablatensor.crn"))
        || jit.has(Category.DRAW_CACHE);

    boolean adj = options.adjoints();
    long start = System.nanoTime();
    Object k = KernelGenerator.generate(tape, adj, roll, segNodes, f32);
    this.compileSeconds = (System.nanoTime() - start) / 1e9;
    this.classFileBytes = KernelGenerator.classFileSize(tape, adj, roll, segNodes, f32);
    this.scratchElems = KernelGenerator.scratchLen(tape, adj, roll);
    this.kernel64 = f32 ? null : (JitKernel) k;
    this.kernel32 = f32 ? (JitKernelF32) k : null;
    if (System.getProperty("nablatensor.jit.debug") != null) {
      System.err.println("[jit] " + KernelGenerator.describeShape(tape, adj, roll)
          + " nodes=" + tape.size() + " " + (f32 ? "fp32" : "fp64")
          + " classBytes=" + classFileBytes
          + " roll=" + roll + " crn=" + crn + " fastMath=" + fastMath);
    }

    this.nodes = tape.size();
    this.vLen = KernelGenerator.vLen(tape, adj, roll);
    this.randCount = tape.randTotal();
    int nStreams = tape.randStreamCount();
    this.streamNormal = new int[nStreams];
    this.streamUniform = new int[nStreams];
    this.streamOffset = new int[nStreams];
    for (int s = 0; s < nStreams; s++) {
      streamNormal[s] = tape.randNormalCount(s);
      streamUniform[s] = tape.randUniformCount(s);
      streamOffset[s] = tape.randStreamOffset(s);
    }
    this.outputNode = new int[tape.outputCount()];
    for (int o = 0; o < outputNode.length; o++) {
      outputNode[o] = KernelGenerator.mapNode(tape, adj, roll, tape.outputNode(o));
    }
    this.inputNode = new int[tape.inputCount()];
    for (int j = 0; j < inputNode.length; j++) {
      inputNode[j] = KernelGenerator.mapNode(tape, adj, roll, tape.inputNode(j));
    }

    this.threads = Math.max(1, options.resolvedThreads());
    this.pool = threads > 1
        ? Executors.newFixedThreadPool(threads, runnable -> {
            Thread thread = new Thread(runnable, "aad-jit");
            thread.setDaemon(true);
            return thread;
          })
        : null;
  }

  @Override
  public String engineName() {
    return "cpu-jit";
  }

  @Override
  public double compileSeconds() {
    return compileSeconds;
  }

  public String describeKernel() {
    return classFileBytes + " B class, seg=" + segNodes + ", " + (f32 ? "fp32" : "fp64");
  }

  @Override
  public AadResult replay(long paths, long pathOffset, long seed) {
    checkOpen();
    if (paths <= 0) {
      throw new IllegalArgumentException("paths must be positive");
    }

    boolean useCrn = crn && randCount > 0 && (long) paths * randCount <= CRN_CAP;
    boolean cacheRead = false;
    boolean cacheWrite = false;
    if (useCrn) {
      boolean hit = cacheFilled && seed == cacheSeed
          && pathOffset == cachePathOffset && paths == cachePaths;
      if (hit) {
        cacheRead = true;
      } else {
        int need = Math.toIntExact(paths * randCount);
        if (f32) {
          if (cache32 == null || cache32.length < need) cache32 = new float[need];
        } else {
          if (cache64 == null || cache64.length < need) cache64 = new double[need];
        }
        cacheSeed = seed;
        cachePathOffset = pathOffset;
        cachePaths = paths;
        cacheFilled = false;
        cacheWrite = true;
      }
    }
    final boolean useCache = cacheRead || cacheWrite;
    final boolean write = cacheWrite;

    long start = System.nanoTime();
    Accumulator total = threads == 1
        ? runRange(pathOffset, paths, seed, useCache, pathOffset, write)
        : runParallel(paths, pathOffset, seed, useCache, write);
    double seconds = (System.nanoTime() - start) / 1e9;

    if (cacheWrite) {
      cacheFilled = true;
    }

    return total.toResult(tape, paths, seconds);
  }

  private Accumulator runParallel(long paths, long pathOffset, long seed, boolean useCache, boolean write) {
    List<Callable<Accumulator>> tasks = new ArrayList<>(threads);
    long each = paths / threads;
    long extra = paths % threads;
    long cursor = pathOffset;
    for (int t = 0; t < threads; t++) {
      long count = each + (t < extra ? 1 : 0);
      long from = cursor;
      cursor += count;
      if (count > 0) {
        tasks.add(() -> runRange(from, count, seed, useCache, pathOffset, write));
      }
    }
    Accumulator total = new Accumulator(outputNode.length, inputNode.length);
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

  private Accumulator runRange(long pathFrom, long count, long seed,
      boolean useCache, long cacheOrigin, boolean write) {
    return f32
        ? runRangeF32(pathFrom, count, seed, useCache, cacheOrigin, write)
        : runRangeF64(pathFrom, count, seed, useCache, cacheOrigin, write);
  }

  private Accumulator runRangeF64(long pathFrom, long count, long seed,
      boolean useCache, long cacheOrigin, boolean write) {
    final double[] v = new double[vLen];
    final double[] d = options.adjoints() ? new double[vLen] : null;
    final double[] draws = new double[Math.max(1, randCount)];
    final double[] scratch = new double[Math.max(1, scratchElems)];
    final double[] in = inputs;
    final boolean adjoints = options.adjoints();
    final boolean read = useCache && !write;
    final int base = useCache ? Math.toIntExact((pathFrom - cacheOrigin) * randCount) : 0;
    final int nOut = outputNode.length;
    final Accumulator acc = new Accumulator(nOut, inputNode.length);

    for (long path = pathFrom; path < pathFrom + count; path++) {
      int slot = base + Math.toIntExact((path - pathFrom) * randCount);
      if (read) {
        System.arraycopy(cache64, slot, draws, 0, randCount);
      } else if (!SKIP_RNG) {
        fillDraws(draws, path, seed);
        if (write) {
          System.arraycopy(draws, 0, cache64, slot, randCount);
        }
      }
      kernel64.forward(v, in, draws, scratch);
      for (int o = 0; o < nOut; o++) {
        double y = v[outputNode[o]];
        acc.value[o] += y;
        acc.sumsq[o] += y * y;
      }
      if (!adjoints) {
        continue;
      }
      for (int o = 0; o < nOut; o++) {
        Arrays.fill(d, 0.0);
        d[outputNode[o]] = 1.0;
        kernel64.reverse(v, d, scratch, draws);
        double[] grow = acc.gradient[o];
        for (int j = 0; j < grow.length; j++) {
          grow[j] += d[inputNode[j]];
        }
      }
    }
    return acc;
  }

  private Accumulator runRangeF32(long pathFrom, long count, long seed,
      boolean useCache, long cacheOrigin, boolean write) {
    final float[] v = new float[vLen];
    final float[] d = options.adjoints() ? new float[vLen] : null;
    final float[] draws = new float[Math.max(1, randCount)];
    final float[] scratch = new float[Math.max(1, scratchElems)];
    final double[] draws64 = new double[Math.max(1, randCount)];
    final double[] in = inputs;
    final float[] inF = new float[in.length];
    final boolean adjoints = options.adjoints();
    final boolean read = useCache && !write;
    final int base = useCache ? Math.toIntExact((pathFrom - cacheOrigin) * randCount) : 0;
    final int nOut = outputNode.length;
    final Accumulator acc = new Accumulator(nOut, inputNode.length);

    for (int j = 0; j < in.length; j++) {
      inF[j] = (float) in[j];
    }

    for (long path = pathFrom; path < pathFrom + count; path++) {
      int slot = base + Math.toIntExact((path - pathFrom) * randCount);
      if (read) {
        System.arraycopy(cache32, slot, draws, 0, randCount);
      } else if (!SKIP_RNG) {
        fillDraws(draws64, path, seed);
        for (int k = 0; k < randCount; k++) {
          draws[k] = (float) draws64[k];
        }
        if (write) {
          System.arraycopy(draws, 0, cache32, slot, randCount);
        }
      }
      kernel32.forward(v, inF, draws, scratch);
      for (int o = 0; o < nOut; o++) {
        double y = v[outputNode[o]];
        acc.value[o] += y;
        acc.sumsq[o] += y * y;
      }
      if (!adjoints) {
        continue;
      }
      for (int o = 0; o < nOut; o++) {
        Arrays.fill(d, 0.0f);
        d[outputNode[o]] = 1.0f;
        kernel32.reverse(v, d, scratch, draws);
        double[] grow = acc.gradient[o];
        for (int j = 0; j < grow.length; j++) {
          grow[j] += d[inputNode[j]];
        }
      }
    }
    return acc;
  }

  @Override
  public void close() {
    super.close();
    if (pool != null) {
      pool.shutdownNow();
    }
  }

  /** Fills the flat per-path draw buffer: each stream's normals then its uniforms. */
  private void fillDraws(double[] draws, long path, long seed) {
    for (int s = 0; s < streamOffset.length; s++) {
      JitPhilox.fillNormals(draws, streamOffset[s], streamNormal[s], path, seed, s, fastMath);
      JitPhilox.fillUniforms(draws, streamOffset[s] + streamNormal[s], streamUniform[s], path, seed, s);
    }
  }

  private static final class Accumulator {
    final double[] value;         // per output: sum over paths of the output value
    final double[] sumsq;        // per output: sum over paths of value^2
    final double[][] gradient;   // [output][input]: sum over paths of the adjoint

    Accumulator(int outputs, int inputs) {
      this.value = new double[outputs];
      this.sumsq = new double[outputs];
      this.gradient = new double[outputs][inputs];
    }

    void add(Accumulator other) {
      for (int o = 0; o < value.length; o++) {
        value[o] += other.value[o];
        sumsq[o] += other.sumsq[o];
        double[] row = gradient[o];
        double[] orow = other.gradient[o];
        for (int j = 0; j < row.length; j++) {
          row[j] += orow[j];
        }
      }
    }

    AadResult toResult(AadTape tape, long paths, double seconds) {
      int no = value.length;
      double[] means = new double[no];
      double[] stderr = new double[no];
      double[][] grads = new double[no][];
      for (int o = 0; o < no; o++) {
        means[o] = value[o] / paths;
        if (paths > 1) {
          double var = (sumsq[o] - paths * means[o] * means[o]) / (paths - 1);
          stderr[o] = Math.sqrt(Math.max(0.0, var) / paths);
        } else {
          stderr[o] = Double.NaN;
        }
        double[] row = gradient[o].clone();
        for (int j = 0; j < row.length; j++) {
          row[j] /= paths;
        }
        grads[o] = row;
      }
      return AadResult.of(tape.outputNames(), means, stderr, grads, tape.inputNames(), paths, seconds);
    }
  }
}
