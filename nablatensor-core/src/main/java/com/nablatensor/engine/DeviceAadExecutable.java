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
package com.nablatensor.engine;

import com.nablatensor.annotation.Internal;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A recorded tape compiled into an accelerator replay kernel, on top of a
 * {@link DeviceRuntime}. One class for every "linear buffer + scalar arguments"
 * backend — CUDA, ROCm, OpenCL — replacing the three near-identical
 * {@code *AadKernel} classes each of those engines used to carry. (The Vulkan
 * engine has its own executable on the same {@link GpuAadExecutable} base:
 * SPIR-V dispatch with push constants and descriptor sets does not fit the
 * {@link DeviceRuntime} surface.)
 *
 * <p>Recording and the kernel compile happen once, in {@link #compile}; after
 * that a {@link #replay} costs one launch plus a small per-work-group partial
 * download, no matter how many scenarios it covers. Inputs are kernel
 * arguments, so {@link #setInput} re-prices a shifted market with no re-record
 * and no re-compile — and an unchanged input set is not re-uploaded.
 *
 * <p>For a long adjoint tape (over {@code nablatensor.checkpoint.minNodes}, and
 * only when {@code -Dnablatensor.checkpoint=on}) the kernel is
 * segment-checkpointed rather than fully unrolled — see {@link AadCheckpointPlan}
 * — which needs a per-invocation scratch buffer and two extra kernel arguments.
 */
@Internal
public final class DeviceAadExecutable extends GpuAadExecutable {

  /** Turns a tape (plus an optional checkpoint plan) into kernel source. */
  @FunctionalInterface
  public interface SourceGenerator {
    String generate(AadTape tape, AadOptions options, AadCheckpointPlan plan);
  }

  /**
   * The portable CUDA-C generator that {@link CudaAadCodegen} emits: the CUDA
   * engine compiles it with NVRTC and the ROCm engine with HIPRTC, both
   * unchanged. The OpenCL engine passes its own dialect rewrite instead.
   */
  public static final SourceGenerator CUDA_C = (tape, options, plan) -> plan != null
      ? CudaAadCodegen.generateCheckpointed(tape, options, plan)
      : CudaAadCodegen.generate(tape, options);

  /**
   * Kernels keyed by {@code engine + compileKey + source}. The source is a pure
   * function of the tape structure and options (input <em>values</em> are
   * arguments), so a book of trades sharing a payoff shape compiles once.
   */
  private static final Map<String, Long> KERNELS = new ConcurrentHashMap<>();

  private final DeviceRuntime runtime;
  private final long kernel;
  private final AadCheckpointPlan plan;
  private final int scratchElemBytes;

  private long inputBuffer;
  private long partialBuffer;
  private int partialBlocks;
  private long scratchBuffer;
  private long scratchInvocations;
  /** The input buffer only needs re-uploading after {@link #setInput}. */
  private boolean inputsDirty = true;
  private boolean closed;

  /**
  * Segment checkpointing is opt-in: numerical parity is precision-dependent,
  * and recompute / checkpoint traffic can outweigh the occupancy gain on some
  * tapes and devices. Enable with
   * {@code -Dnablatensor.checkpoint=on} or an explicit
   * {@code -Dnablatensor.checkpoint.minNodes=<n>}.
   */
  public static int checkpointMinNodes() {
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

  /**
   * Record-once compile. {@code maxChunkSeconds} is the per-dispatch wall-clock
   * budget the base class uses to size chunks — tighter on a GPU that also
   * drives a display.
   */
  public static DeviceAadExecutable compile(AadTape tape, AadOptions options, String engineName,
                                            DeviceRuntime runtime, SourceGenerator sources,
                                            double maxChunkSeconds) {
    AadCheckpointPlan plan = AadCheckpointPlan.of(tape, options, checkpointMinNodes());
    String source = sources.generate(tape, options, plan);
    String cacheKey = engineName + '\0' + runtime.compileKey() + '\0' + source;
    long start = System.nanoTime();
    long kernel = KERNELS.computeIfAbsent(cacheKey,
        key -> runtime.compile(source, CudaAadCodegen.KERNEL_NAME));
    double seconds = (System.nanoTime() - start) / 1e9;
    return new DeviceAadExecutable(tape, options, engineName, runtime, kernel, seconds, plan,
        maxChunkSeconds);
  }

  private DeviceAadExecutable(AadTape tape, AadOptions options, String engineName,
                              DeviceRuntime runtime, long kernel, double compileSeconds,
                              AadCheckpointPlan plan, double maxChunkSeconds) {
    super(tape, options, engineName, compileSeconds);
    this.runtime = runtime;
    this.kernel = kernel;
    this.plan = plan;
    this.scratchElemBytes = options.precision() == AadOptions.Precision.FLOAT32
        ? Float.BYTES : Double.BYTES;
    setMaxChunkSeconds(maxChunkSeconds);
  }

  @Override
  public void setInput(String name, double value) {
    super.setInput(name, value);
    inputsDirty = true;
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
    if (inputsDirty) {
      runtime.uploadDoubles(inputBuffer, inputs);
      inputsDirty = false;
    }

    long start = System.nanoTime();
    if (plan != null) {
      runtime.launch(kernel, blocks, CudaAadCodegen.BLOCK,
          inputBuffer, paths, pathOffset, seed, partialBuffer, scratchBuffer, invocations);
    } else {
      runtime.launch(kernel, blocks, CudaAadCodegen.BLOCK,
          inputBuffer, paths, pathOffset, seed, partialBuffer);
    }
    runtime.synchronize();
    double[] partials = runtime.downloadDoubles(partialBuffer, blocks * channels);
    double seconds = (System.nanoTime() - start) / 1e9;

    double[] sums = newSums();
    accumulate(partials, blocks, sums);
    return finish(sums, paths, seconds);
  }

  private void ensureBuffers(int blocks, long invocations) {
    if (inputBuffer == 0) {
      inputBuffer = runtime.malloc((long) Math.max(1, inputs.length) * Double.BYTES);
    }
    if (blocks > partialBlocks) {
      if (partialBuffer != 0) {
        runtime.free(partialBuffer);
      }
      partialBuffer = runtime.malloc((long) blocks * channels * Double.BYTES);
      partialBlocks = blocks;
    }
    if (plan != null && invocations > scratchInvocations) {
      if (scratchBuffer != 0) {
        runtime.free(scratchBuffer);
      }
      scratchBuffer = runtime.malloc(invocations * plan.slotsPerPath * scratchElemBytes);
      scratchInvocations = invocations;
    }
  }

  /**
   * Work-items persist across scenarios through a grid-stride loop, so the grid
   * is capped rather than scaled with the scenario count. The checkpointed
   * kernel uses a tighter cap because it also sizes a per-invocation scratch
   * buffer.
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
      runtime.free(inputBuffer);
      inputBuffer = 0;
    }
    if (partialBuffer != 0) {
      runtime.free(partialBuffer);
      partialBuffer = 0;
    }
    if (scratchBuffer != 0) {
      runtime.free(scratchBuffer);
      scratchBuffer = 0;
    }
  }
}
