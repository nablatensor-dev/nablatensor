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
package com.nablatensor.engine.rocm;

import com.nablatensor.engine.AadCheckpointPlan;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.AbstractAadExecutable;
import com.nablatensor.engine.CudaAadCodegen;
import com.nablatensor.backend.rocm.HipCompute;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A recorded tape compiled into a HIP replay kernel.
 *
 * <p>Recording and HIPRTC compilation happen once, in {@link #compile}; after
 * that a {@link #replay} costs one {@code hipModuleLaunchKernel} plus a small
 * block-partial download, no matter how many scenarios it covers. The generated
 * source is {@link CudaAadCodegen}'s portable CUDA C — HIPRTC takes it
 * unchanged — and inputs are kernel arguments, so {@link #setInput} re-prices a
 * shifted market with no re-record and no re-compile.
 *
 * <p>For a long adjoint tape (over {@code nablatensor.checkpoint.minNodes}, 512
 * by default; disable with {@code -Dnablatensor.checkpoint=off}) the kernel is
 * segment-checkpointed instead of fully unrolled — see {@link AadCheckpointPlan}
 * — which needs a per-invocation scratch buffer and two extra kernel arguments.
 */
public final class RocmAadKernel extends AbstractAadExecutable {

  private static final Map<String, Long> KERNELS = new ConcurrentHashMap<>();

  private final long function;
  private final int channels;
  private final double compileSeconds;
  private final AadCheckpointPlan plan;
  private final int elemBytes;

  private long inputBuffer;
  private long partialBuffer;
  private int partialBlocks;
  private long scratchBuffer;
  private long scratchInvocations;
  private boolean closed;

  private RocmAadKernel(AadTape tape, AadOptions options, long function, double compileSeconds,
                         AadCheckpointPlan plan) {
    super(tape, options);
    this.function = function;
    this.channels = options.adjoints() ? tape.inputCount() + 1 : 1;
    this.compileSeconds = compileSeconds;
    this.plan = plan;
    this.elemBytes = options.precision() == AadOptions.Precision.FLOAT32 ? Float.BYTES : Double.BYTES;
  }

  /**
   * Segment checkpointing is <em>opt-in</em>: it is implemented and verified
   * bit-for-bit against the unrolled kernel, but on the integrated GPU this
   * repo runs on it is a wash-to-slightly-slower — the forward recompute (RNG
   * heavy) and the checkpoint-buffer traffic cost as much as the occupancy it
   * buys back on a ~120 GB/s APU. It is expected to pay on a datacenter GPU
   * (large register file, ample HBM bandwidth). Enable with
   * {@code -Dnablatensor.checkpoint=on} or an explicit
   * {@code -Dnablatensor.checkpoint.minNodes=<n>}.
   */
  static int checkpointMinNodes() {
    String mode = System.getProperty("nablatensor.checkpoint", "");
    if ("on".equalsIgnoreCase(mode)) {
      return Integer.getInteger("nablatensor.checkpoint.minNodes", 512);
    }
    if ("off".equalsIgnoreCase(mode)) {
      return Integer.MAX_VALUE;
    }
    return System.getProperty("nablatensor.checkpoint.minNodes") != null
        ? Integer.getInteger("nablatensor.checkpoint.minNodes")
        : Integer.MAX_VALUE;
  }

  public static RocmAadKernel compile(AadTape tape, AadOptions options) {
    if (!HipCompute.isAvailable()) {
      throw new IllegalStateException("no ROCm/HIP device available for the AAD replay kernel");
    }
    AadCheckpointPlan plan = AadCheckpointPlan.of(tape, options, checkpointMinNodes());
    String source = plan != null
        ? CudaAadCodegen.generateCheckpointed(tape, options, plan)
        : CudaAadCodegen.generate(tape, options);
    long start = System.nanoTime();
    long function = KERNELS.computeIfAbsent(source,
        key -> HipCompute.compile(key, CudaAadCodegen.KERNEL_NAME));
    double seconds = (System.nanoTime() - start) / 1e9;
    return new RocmAadKernel(tape, options, function, seconds, plan);
  }

  @Override
  public String engineName() {
    return "rocm";
  }

  @Override
  public double compileSeconds() {
    return compileSeconds;
  }

  /**
   * An integrated GPU shares the DRM scheduler with the display; a wedged
   * kernel there can take the driver down, so a dispatch is kept well below the
   * seconds-range hang check. Override with {@code -Dnablatensor.maxLaunchSeconds}.
   */
  @Override
  protected double defaultMaxChunkSeconds() {
    String override = System.getProperty("nablatensor.maxLaunchSeconds");
    return override != null ? Double.parseDouble(override) : 0.5;
  }

  @Override
  public AadResult replay(long paths, long pathOffset, long seed) {
    checkOpen();
    if (paths <= 0) {
      throw new IllegalArgumentException("paths must be positive");
    }
    int blocks = grid(paths);
    long invocations = (long) blocks * CudaAadCodegen.BLOCK;
    ensureBuffers(blocks, invocations);
    HipCompute.uploadDoubles(inputBuffer, inputs);

    long start = System.nanoTime();
    if (plan != null) {
      HipCompute.launch(function, blocks, CudaAadCodegen.BLOCK,
          inputBuffer, paths, pathOffset, seed, partialBuffer, scratchBuffer, invocations);
    } else {
      HipCompute.launch(function, blocks, CudaAadCodegen.BLOCK,
          inputBuffer, paths, pathOffset, seed, partialBuffer);
    }
    HipCompute.synchronize();
    double seconds = (System.nanoTime() - start) / 1e9;

    double[] partials = HipCompute.downloadDoubles(partialBuffer, blocks * channels);
    double value = 0.0;
    double[] gradients = new double[tape.inputCount()];
    for (int block = 0; block < blocks; block++) {
      value += partials[block * channels];
      for (int c = 1; c < channels; c++) {
        gradients[c - 1] += partials[block * channels + c];
      }
    }
    value /= paths;
    for (int j = 0; j < gradients.length; j++) {
      gradients[j] /= paths;
    }
    return new AadResult(value, gradients, tape.inputNames(), paths, seconds);
  }

  private void ensureBuffers(int blocks, long invocations) {
    if (inputBuffer == 0) {
      inputBuffer = HipCompute.malloc((long) Math.max(1, inputs.length) * Double.BYTES);
    }
    if (blocks > partialBlocks) {
      if (partialBuffer != 0) {
        HipCompute.free(partialBuffer);
      }
      partialBuffer = HipCompute.malloc((long) blocks * channels * Double.BYTES);
      partialBlocks = blocks;
    }
    if (plan != null && invocations > scratchInvocations) {
      if (scratchBuffer != 0) {
        HipCompute.free(scratchBuffer);
      }
      scratchBuffer = HipCompute.malloc(invocations * plan.slotsPerPath * elemBytes);
      scratchInvocations = invocations;
    }
  }

  /**
   * Threads persist across scenarios through a grid-stride loop, so the grid is
   * capped rather than scaled with the scenario count. The checkpointed kernel
   * uses a tighter cap because it also sizes a per-invocation scratch buffer.
   */
  private int grid(long paths) {
    long needed = (paths + CudaAadCodegen.BLOCK - 1) / CudaAadCodegen.BLOCK;
    int cap = plan != null ? 1024 : 4096;
    return (int) Math.max(1, Math.min(needed, cap));
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    super.close();
    if (inputBuffer != 0) {
      HipCompute.free(inputBuffer);
      inputBuffer = 0;
    }
    if (partialBuffer != 0) {
      HipCompute.free(partialBuffer);
      partialBuffer = 0;
    }
    if (scratchBuffer != 0) {
      HipCompute.free(scratchBuffer);
      scratchBuffer = 0;
    }
  }
}
