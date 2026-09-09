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
import com.nablatensor.engine.AbstractAadExecutable;
import com.nablatensor.backend.vulkan.VulkanCompute;

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
 * market with no re-record and no re-compile.
 */
public final class VulkanAadKernel extends AbstractAadExecutable {

  private static final int MAX_GROUPS = Integer.getInteger("nablatensor.vulkan.groups", 2048);

  /** Pipelines are keyed by generated source, so one payoff shape compiles once. */
  private static final Map<String, String> PIPELINES = new ConcurrentHashMap<>();

  private final String kernelName;
  private final int channels;
  private final double compileSeconds;

  private long inputBuffer;
  private long inputMemory;
  private long partialBuffer;
  private long partialMemory;
  private boolean closed;

  private VulkanAadKernel(AadTape tape, AadOptions options, String kernelName,
                           double compileSeconds) {
    super(tape, options);
    this.kernelName = kernelName;
    this.channels = options.adjoints() ? tape.inputCount() + 1 : 1;
    this.compileSeconds = compileSeconds;
    long[] in = VulkanCompute.alloc((long) Math.max(1, tape.inputCount()) * Float.BYTES);
    this.inputBuffer = in[0];
    this.inputMemory = in[1];
    long[] part = VulkanCompute.alloc((long) MAX_GROUPS * channels * Float.BYTES);
    this.partialBuffer = part[0];
    this.partialMemory = part[1];
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
    String source = VulkanAadCodegen.generate(tape, options);
    String name = "aad_" + Integer.toHexString(source.hashCode());
    long start = System.nanoTime();
    PIPELINES.computeIfAbsent(source, key -> {
      VulkanCompute.registerPipeline(name, key, 2);
      return name;
    });
    double seconds = (System.nanoTime() - start) / 1e9;
    return new VulkanAadKernel(tape, options, name, seconds);
  }

  @Override
  public String engineName() {
    return "vulkan";
  }

  @Override
  public double compileSeconds() {
    return compileSeconds;
  }

  /**
   * RADV on a shared-memory APU has no display-driver watchdog as aggressive as
   * NVIDIA's TDR, but the kernel DRM scheduler still has a hang check in the
   * seconds range, so a dispatch is kept about an order of magnitude below it.
   */
  @Override
  protected double defaultMaxChunkSeconds() {
    String override = System.getProperty("nablatensor.maxLaunchSeconds");
    return override != null ? Double.parseDouble(override) : 1.0;
  }

  @Override
  public AadResult replay(long paths, long pathOffset, long seed) {
    checkOpen();
    if (paths <= 0) {
      throw new IllegalArgumentException("paths must be positive");
    }

    float[] inF = new float[Math.max(1, inputs.length)];
    for (int j = 0; j < inputs.length; j++) {
      inF[j] = (float) inputs[j];
    }
    VulkanCompute.writeFloats(inputMemory, inF);

    int seedLo = (int) seed;
    int seedHi = (int) (seed >>> 32);
    long[] buffers = {inputBuffer, partialBuffer};

    double value = 0.0;
    double[] gradients = new double[tape.inputCount()];

    long start = System.nanoTime();
    long done = 0;
    while (done < paths) {
      int sub = (int) Math.min((long) VulkanAadCodegen.MAX_DISPATCH_PATHS, paths - done);
      int groups = (int) Math.min((long) MAX_GROUPS,
          Math.max(1L, (sub + VulkanAadCodegen.LOCAL_SIZE - 1L) / VulkanAadCodegen.LOCAL_SIZE));
      long base = pathOffset + done;
      int[] push = {sub, (int) base, (int) (base >>> 32), seedLo, seedHi};

      VulkanCompute.dispatch(kernelName, groups, buffers, push);

      float[] partials = VulkanCompute.readFloats(partialMemory, groups * channels);
      for (int g = 0; g < groups; g++) {
        value += partials[g * channels];
        for (int c = 1; c < channels; c++) {
          gradients[c - 1] += partials[g * channels + c];
        }
      }
      done += sub;
    }
    double seconds = (System.nanoTime() - start) / 1e9;

    value /= paths;
    for (int j = 0; j < gradients.length; j++) {
      gradients[j] /= paths;
    }
    return new AadResult(value, gradients, tape.inputNames(), paths, seconds);
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
