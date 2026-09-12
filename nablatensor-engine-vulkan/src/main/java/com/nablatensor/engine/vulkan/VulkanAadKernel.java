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
package com.nablatensor.engine.vulkan;

import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.GpuAadExecutable;
import com.nablatensor.backend.vulkan.VulkanCompute;
import com.nablatensor.engine.vulkan.VulkanAadCodegen.Config;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A recorded tape compiled into a Vulkan compute pipeline and replayed one
 * scenario per invocation.
 *
 * <p>Recording and SPIR-V compilation happen once, in {@link #compile}; after
 * that a {@link #replay} costs one (or, for very large batches, a handful of)
 * {@code vkCmdDispatch} submissions plus a small per-workgroup read-back, no
 * matter how many scenarios it covers. Inputs are storage-buffer contents
 * rather than baked-in constants, so {@link #setInput} re-prices a shifted
 * market with no re-record and no re-compile, and {@link #replayMany} prices
 * a whole ladder of markets from one dispatch.
 */
public final class VulkanAadKernel extends GpuAadExecutable {

  private static final int MAX_GROUPS = Integer.getInteger("nablatensor.vulkan.groups", 2048);

  /** Pipelines are keyed by generated source, so one payoff shape compiles once. */
  private static final Map<String, String> PIPELINES = new ConcurrentHashMap<>();

  private final Config config;
  private final String kernelName;
  /** Multi-market variants of this kernel, by market count; compiled on first use. */
  private final Map<Integer, String> ladderKernels = new HashMap<>();

  private long inputBuffer;
  private long inputMemory;
  private int inputCapacity;   // floats
  private long partialBuffer;
  private long partialMemory;
  private int partialCapacity; // floats
  private boolean closed;

  private VulkanAadKernel(AadTape tape, AadOptions options, Config config, String kernelName,
                           double compileSeconds) {
    super(tape, options, "vulkan", compileSeconds);
    this.config = config;
    this.kernelName = kernelName;
    ensureInputs(Math.max(1, tape.inputCount()));
    ensurePartials(MAX_GROUPS * channels);
  }

  private void ensureInputs(int floats) {
    if (inputBuffer != 0 && inputCapacity >= floats) {
      return;
    }
    if (inputBuffer != 0) {
      VulkanCompute.free(inputBuffer, inputMemory);
    }
    long[] in = VulkanCompute.alloc((long) floats * Float.BYTES);
    inputBuffer = in[0];
    inputMemory = in[1];
    inputCapacity = floats;
  }

  private void ensurePartials(int floats) {
    if (partialBuffer != 0 && partialCapacity >= floats) {
      return;
    }
    if (partialBuffer != 0) {
      VulkanCompute.free(partialBuffer, partialMemory);
    }
    long[] part = VulkanCompute.alloc((long) floats * Float.BYTES);
    partialBuffer = part[0];
    partialMemory = part[1];
    partialCapacity = floats;
  }

  public static boolean vulkanAvailable() {
    try {
      return VulkanCompute.isAvailable();
    } catch (Throwable ignored) {
      return false;
    }
  }

  public static String vulkanDeviceName() {
    try {
      return VulkanCompute.deviceName();
    } catch (Throwable ignored) {
      return "no device";
    }
  }

  public static VulkanAadKernel compile(AadTape tape, AadOptions options) {
    if (options.precision() != AadOptions.Precision.FLOAT32) {
      throw new IllegalArgumentException("the Vulkan AAD engine is single-precision only");
    }
    if (!vulkanAvailable()) {
      throw new IllegalStateException("no Vulkan compute device available for the AAD replay kernel");
    }
    Config config = Config.forTape(tape, options);
    long start = System.nanoTime();
    String name = register(tape, options, config);
    double seconds = (System.nanoTime() - start) / 1e9;
    return new VulkanAadKernel(tape, options, config, name, seconds);
  }

  private static String register(AadTape tape, AadOptions options, Config config) {
    String source = VulkanAadCodegen.generate(tape, options, config);
    String name = "aad_" + Integer.toHexString(source.hashCode());
    String dump = System.getProperty("nablatensor.vulkan.dump");
    if (dump != null) {
      try {
        Files.writeString(Path.of(dump, name + ".comp"), source);
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    }
    return PIPELINES.computeIfAbsent(source, key -> {
      VulkanCompute.registerPipeline(name, key, 2);
      return name;
    });
  }

  /**
   * RADV on a shared-memory APU has no display-driver watchdog as aggressive as
   * NVIDIA's TDR, but the kernel DRM scheduler still has a hang check in the
   * seconds range, so a dispatch is kept about an order of magnitude below it.
   */
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
   * order) over the same paths and seed in one dispatch. Each path's draws are
   * generated once and shared across the sets in registers, so a revaluation
   * ladder pays for its random numbers once rather than once per market — the
   * GPU's form of a draw cache, with nothing written to memory. Results are
   * bit-identical to {@code inputSets.length} separate {@link #replay} calls.
   * A variant of the pipeline is compiled for each distinct set count on first
   * use; register pressure grows with the count, so a handful of markets per
   * call is the sweet spot for long tapes.
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
    VulkanCompute.writeFloats(inputMemory, inF);

    int seedLo = (int) seed;
    int seedHi = (int) (seed >>> 32);
    long[] buffers = {inputBuffer, partialBuffer};

    int slots = channels * markets;
    double[] sums = new double[slots];

    long start = System.nanoTime();
    long done = 0;
    while (done < paths) {
      int sub = (int) Math.min((long) VulkanAadCodegen.MAX_DISPATCH_PATHS, paths - done);
      int groups = (int) Math.min((long) MAX_GROUPS,
          Math.max(1L, (sub + VulkanAadCodegen.LOCAL_SIZE - 1L) / VulkanAadCodegen.LOCAL_SIZE));
      long base = pathOffset + done;
      int[] push = new int[VulkanAadCodegen.PC_COUNT];
      push[VulkanAadCodegen.PC_N_LOCAL] = sub;
      push[VulkanAadCodegen.PC_OFF_LO] = (int) base;
      push[VulkanAadCodegen.PC_OFF_HI] = (int) (base >>> 32);
      push[VulkanAadCodegen.PC_SEED_LO] = seedLo;
      push[VulkanAadCodegen.PC_SEED_HI] = seedHi;
      push[VulkanAadCodegen.PC_ZERO] = 0;

      VulkanCompute.dispatch(kernel, groups, buffers, push);

      float[] partials = VulkanCompute.readFloats(partialMemory, groups * slots);
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
      VulkanCompute.free(inputBuffer, inputMemory);
      inputBuffer = 0;
    }
    if (partialBuffer != 0) {
      VulkanCompute.free(partialBuffer, partialMemory);
      partialBuffer = 0;
    }
  }
}
