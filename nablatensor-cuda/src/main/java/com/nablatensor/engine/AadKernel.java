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

import com.nablatensor.backend.cuda.CudaJit;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A recorded tape compiled into a CUDA replay kernel.
 *
 * <p>Recording and NVRTC compilation happen once, in {@link #compile}; after
 * that a {@link #replay} costs one kernel launch plus a small block-partial
 * download, no matter how many scenarios it covers. Inputs can be changed
 * between replays via {@link #setInput} without re-recording or re-compiling,
 * since they are kernel arguments rather than baked-in constants.
 */
public final class AadKernel extends AbstractAadExecutable {

  private final String source;
  private final long function;
  private final int channels;
  private final double compileSeconds;

  private long inputBuffer;
  private long partialBuffer;
  private int partialBlocks;
  private boolean closed;

  private AadKernel(AadTape tape, AadOptions options, String source, long function,
                     double compileSeconds) {
    super(tape, options);
    this.source = source;
    this.function = function;
    this.channels = options.adjoints() ? tape.inputCount() + 1 : 1;
    this.compileSeconds = compileSeconds;
  }

  @Override
  public String engineName() {
    return "cuda";
  }

  /**
   * The GPU is the one backend where an overlong dispatch is actively
   * dangerous: when the device also drives a display the driver's
   * {@code KERNEL_EXEC_TIMEOUT} watchdog recovers by resetting the GPU, taking
   * the screen down with it, so the budget there is an order of magnitude below
   * the watchdog rather than merely convenient.
   */
  @Override
  protected double defaultMaxChunkSeconds() {
    String override = System.getProperty("nablatensor.maxLaunchSeconds");
    if (override != null) {
      return Double.parseDouble(override);
    }
    return CudaJit.kernelTimeoutEnabled() ? 0.2 : 2.0;
  }

  public static boolean cudaAvailable() {
    return CudaJit.isAvailable();
  }

  public static String cudaDeviceName() {
    return CudaJit.deviceName();
  }

  public static AadKernel compile(AadTape tape) {
    return compile(tape, AadOptions.defaults());
  }

  /**
   * Compiled kernels keyed by their generated source.
   *
   * <p>The source is the exact cache key by construction: it is a pure function
   * of the tape's structure and the options, and it deliberately does not
   * contain the input values, which are kernel arguments. So a book of trades
   * sharing one payoff shape, or the same tape re-recorded after a restart of
   * the pricing loop, compiles once rather than once per instance. Recorded
   * here it removes seconds per repeat; the same map written to disk would
   * remove them across processes too.
   */
  private static final Map<String, Long> KERNELS = new ConcurrentHashMap<>();

  public static AadKernel compile(AadTape tape, AadOptions options) {
    if (!CudaJit.isAvailable()) {
      throw new IllegalStateException("no CUDA device available for the AAD replay kernel");
    }
    String source = CudaAadCodegen.generate(tape, options);
    long start = System.nanoTime();
    Long cached = KERNELS.get(source);
    long function = cached != null
        ? cached
        : KERNELS.computeIfAbsent(source, key -> CudaJit.compile(key, CudaAadCodegen.KERNEL_NAME));
    double seconds = (System.nanoTime() - start) / 1e9;
    return new AadKernel(tape, options, source, function, seconds);
  }

  /** Generated CUDA source, handy for inspecting what the recording produced. */
  public String source() {
    return source;
  }

  /** Wall-clock NVRTC compilation time, paid once per kernel. */
  @Override
  public double compileSeconds() {
    return compileSeconds;
  }

  @Override
  public AadResult replay(long paths, long pathOffset, long seed) {
    checkOpen();
    if (paths <= 0) {
      throw new IllegalArgumentException("paths must be positive");
    }
    int blocks = grid(paths);
    ensureBuffers(blocks);
    CudaJit.uploadDoubles(inputBuffer, inputs);

    long start = System.nanoTime();
    CudaJit.launch(function, blocks, CudaAadCodegen.BLOCK,
        inputBuffer, paths, pathOffset, seed, partialBuffer);
    CudaJit.synchronize();
    double seconds = (System.nanoTime() - start) / 1e9;

    double[] partials = CudaJit.downloadDoubles(partialBuffer, blocks * channels);
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

  private void ensureBuffers(int blocks) {
    if (inputBuffer == 0) {
      inputBuffer = CudaJit.malloc((long) Math.max(1, inputs.length) * Double.BYTES);
    }
    if (blocks > partialBlocks) {
      if (partialBuffer != 0) {
        CudaJit.free(partialBuffer);
      }
      partialBuffer = CudaJit.malloc((long) blocks * channels * Double.BYTES);
      partialBlocks = blocks;
    }
  }

  /**
   * Threads persist across scenarios through a grid-stride loop, so the grid is
   * capped rather than scaled with the scenario count: the whole per-thread
   * register state (which for an unrolled adjoint sweep is the expensive part)
   * is then set up once per thread instead of once per scenario.
   */
  private static int grid(long paths) {
    long needed = (paths + CudaAadCodegen.BLOCK - 1) / CudaAadCodegen.BLOCK;
    return (int) Math.max(1, Math.min(needed, 4096));
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    super.close();
    if (inputBuffer != 0) {
      CudaJit.free(inputBuffer);
      inputBuffer = 0;
    }
    if (partialBuffer != 0) {
      CudaJit.free(partialBuffer);
      partialBuffer = 0;
    }
  }
}
