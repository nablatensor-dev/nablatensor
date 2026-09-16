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
package com.nablatensor.engine.wgpu;

import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.GpuAadExecutable;
import com.nablatensor.backend.wgpu.WgpuCompute;
import com.nablatensor.engine.wgpu.WgpuAadCodegen.Config;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A recorded tape compiled into a wgpu compute pipeline and replayed one
 * scenario per invocation. The WebGPU counterpart of {@code VulkanAadKernel};
 * see {@code WgpuRuntime} for why each dispatch here submits and waits
 * individually instead of batching several before a submit — this caller
 * always reads partials back right after a dispatch anyway, so nothing is
 * lost by keeping it simple.
 */
public final class WgpuAadKernel extends GpuAadExecutable {

  /** Read-only/read-write shape of the 3 bindings every AAD kernel here uses. */
  private static final boolean[] BINDING_READ_ONLY = {true, false, true};

  /** Pipelines are keyed by generated source, so one payoff shape compiles once. */
  private static final Map<String, String> PIPELINES = new ConcurrentHashMap<>();

  private final Config config;
  private final String kernelName;
  /** Multi-market variants of this kernel, by market count; compiled on first use. */
  private final Map<Integer, String> ladderKernels = new HashMap<>();

  private long inputBuffer;
  private int inputCapacity;    // floats
  private long partialBuffer;
  private int partialCapacity;  // floats
  private final long pushBuffer;
  private boolean closed;

  private WgpuAadKernel(AadTape tape, AadOptions options, Config config, String kernelName,
                         double compileSeconds) {
    super(tape, options, "wgpu", compileSeconds);
    this.config = config;
    this.kernelName = kernelName;
    ensureInputs(Math.max(1, tape.inputCount()));
    ensurePartials(MAX_GROUPS * channels);
    this.pushBuffer = WgpuCompute.alloc((long) WgpuAadCodegen.PC_COUNT * Integer.BYTES);
  }

  private static final int MAX_GROUPS = Integer.getInteger("nablatensor.wgpu.groups", 2048);

  private void ensureInputs(int floats) {
    if (inputBuffer != 0 && inputCapacity >= floats) {
      return;
    }
    if (inputBuffer != 0) {
      WgpuCompute.free(inputBuffer);
    }
    inputBuffer = WgpuCompute.alloc((long) floats * Float.BYTES);
    inputCapacity = floats;
  }

  private void ensurePartials(int floats) {
    if (partialBuffer != 0 && partialCapacity >= floats) {
      return;
    }
    if (partialBuffer != 0) {
      WgpuCompute.free(partialBuffer);
    }
    partialBuffer = WgpuCompute.alloc((long) floats * Float.BYTES);
    partialCapacity = floats;
  }

  public static boolean wgpuAvailable() {
    try {
      return WgpuCompute.isAvailable();
    } catch (Throwable ignored) {
      return false;
    }
  }

  public static String wgpuDeviceName() {
    try {
      return WgpuCompute.deviceName();
    } catch (Throwable ignored) {
      return "no device";
    }
  }

  public static WgpuAadKernel compile(AadTape tape, AadOptions options) {
    if (options.precision() != AadOptions.Precision.FLOAT32) {
      throw new IllegalArgumentException("the wgpu AAD engine is single-precision only");
    }
    if (!wgpuAvailable()) {
      throw new IllegalStateException("no WebGPU device available for the AAD replay kernel");
    }
    Config config = Config.forTape(tape, options);
    long start = System.nanoTime();
    String name = register(tape, options, config);
    double seconds = (System.nanoTime() - start) / 1e9;
    return new WgpuAadKernel(tape, options, config, name, seconds);
  }

  private static String register(AadTape tape, AadOptions options, Config config) {
    String source = WgpuAadCodegen.generate(tape, options, config);
    String name = "aad_" + Integer.toHexString(source.hashCode());
    String dump = System.getProperty("nablatensor.wgpu.dump");
    if (dump != null) {
      try {
        Files.writeString(Path.of(dump, name + ".wgsl"), source);
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    }
    return PIPELINES.computeIfAbsent(source, key -> {
      WgpuCompute.registerPipeline(name, key, BINDING_READ_ONLY);
      return name;
    });
  }

  @Override
  protected double defaultMaxChunkSeconds() {
    return maxLaunchSeconds(1.0);
  }

  @Override
  public AadResult replay(long paths, long pathOffset, long seed) {
    return replayMany(new double[][] {inputs}, paths, pathOffset, seed)[0];
  }

  /**
   * Prices every input set in {@code inputSets} (each in {@link AadTape#inputNames()}
   * order) over the same paths and seed, one dispatch per chunk of paths. See
   * {@code VulkanAadKernel#replayMany} for the draw-cache rationale; the
   * numerics here are bit-identical to it (same tape, same Philox, same Kahan
   * compensation), only the dispatch mechanics differ.
   */
  public AadResult[] replayMany(double[][] inputSets, long paths, long pathOffset, long seed) {
    checkOpen();
    if (paths <= 0) {
      throw new IllegalArgumentException("paths must be positive");
    }
    int markets = inputSets.length;
    if (markets == 0) {
      throw new IllegalArgumentException("at least one input set is required");
    }
    String kernel = markets == 1 ? kernelName
        : ladderKernels.computeIfAbsent(markets, m -> register(tape, options, config.withMarkets(m)));
    int nIn = inputs.length;
    float[] inF = new float[Math.max(1, markets * nIn)];
    for (int m = 0; m < markets; m++) {
      if (inputSets[m].length != nIn) {
        throw new IllegalArgumentException("input set " + m + " has " + inputSets[m].length
            + " values; the tape has " + nIn + " inputs");
      }
      for (int j = 0; j < nIn; j++) {
        inF[m * nIn + j] = (float) inputSets[m][j];
      }
    }
    ensureInputs(inF.length);
    ensurePartials(MAX_GROUPS * channels * markets);
    WgpuCompute.writeFloats(inputBuffer, inF);

    int seedLo = (int) seed;
    int seedHi = (int) (seed >>> 32);
    long[] buffers = {inputBuffer, partialBuffer, pushBuffer};

    int slots = channels * markets;
    double[] sums = new double[slots];

    long start = System.nanoTime();
    long done = 0;
    while (done < paths) {
      int sub = (int) Math.min((long) WgpuAadCodegen.MAX_DISPATCH_PATHS, paths - done);
      int groups = (int) Math.min((long) MAX_GROUPS,
          Math.max(1L, (sub + WgpuAadCodegen.LOCAL_SIZE - 1L) / WgpuAadCodegen.LOCAL_SIZE));
      long base = pathOffset + done;
      int[] push = new int[WgpuAadCodegen.PC_COUNT];
      push[WgpuAadCodegen.PC_N_LOCAL] = sub;
      push[WgpuAadCodegen.PC_OFF_LO] = (int) base;
      push[WgpuAadCodegen.PC_OFF_HI] = (int) (base >>> 32);
      push[WgpuAadCodegen.PC_SEED_LO] = seedLo;
      push[WgpuAadCodegen.PC_SEED_HI] = seedHi;
      push[WgpuAadCodegen.PC_ZERO] = 0;

      WgpuCompute.dispatch(kernel, groups, buffers, push);

      float[] partials = WgpuCompute.readFloats(partialBuffer, groups * slots);
      for (int g = 0; g < groups; g++) {
        for (int c = 0; c < slots; c++) {
          sums[c] += partials[g * slots + c];
        }
      }
      done += sub;
    }
    double seconds = (System.nanoTime() - start) / 1e9;

    AadResult[] results = new AadResult[markets];
    for (int m = 0; m < markets; m++) {
      double[] part = new double[channels];
      System.arraycopy(sums, m * channels, part, 0, channels);
      results[m] = finish(part, paths, seconds);
    }
    return results;
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    super.close();
    if (inputBuffer != 0) {
      WgpuCompute.free(inputBuffer);
      inputBuffer = 0;
    }
    if (partialBuffer != 0) {
      WgpuCompute.free(partialBuffer);
      partialBuffer = 0;
    }
    WgpuCompute.free(pushBuffer);
  }
}
