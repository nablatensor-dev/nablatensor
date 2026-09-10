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

/**
 * The half of an engine that every accelerator replay shares: the channel
 * layout of the per-work-group partial buffer, the reduction of those partials
 * into a value and a gradient, and the dispatch-length budget.
 *
 * <p>Each work-group writes {@code channels} doubles (floats on Vulkan): the
 * summed output value for the scenarios it covered, followed by the summed
 * adjoint of each differentiable input. The host adds those up and divides by
 * the scenario count. That layout is the same whether the dispatch came from
 * CUDA, HIP, OpenCL or a SPIR-V pipeline, so it lives here rather than in each
 * of them.
 *
 * <p>{@link DeviceAadExecutable} covers the three "linear buffer + kernel
 * arguments" runtimes on top of a {@link DeviceRuntime}; the Vulkan engine
 * extends this directly, because descriptor sets and push constants do not fit
 * that surface. The host counterpart is {@link HostAadExecutable}.
 *
 * <p>Public only so the engine modules can extend it; not part of the supported
 * API.
 */
@Internal
public abstract class GpuAadExecutable extends AbstractAadExecutable {

  private final String engineName;
  private final double compileSeconds;

  /** Partial-buffer stride: the value, plus one adjoint per input when asked for. */
  protected final int channels;

  protected GpuAadExecutable(AadTape tape, AadOptions options, String engineName,
                             double compileSeconds) {
    super(tape, options);
    this.engineName = engineName;
    this.compileSeconds = compileSeconds;
    this.channels = options.adjoints() ? tape.inputCount() + 1 : 1;
  }

  @Override
  public final String engineName() {
    return engineName;
  }

  @Override
  public final double compileSeconds() {
    return compileSeconds;
  }

  /**
   * Running sums across however many dispatches one {@link #replay} took:
   * {@code [0]} is the output value and {@code [1 + j]} the adjoint of input
   * {@code j}, matching the partial-buffer channel order.
   */
  protected final double[] newSums() {
    return new double[channels];
  }

  /** Adds the {@code groups} partial records at the front of {@code partials}. */
  protected final void accumulate(double[] partials, int groups, double[] sums) {
    for (int g = 0; g < groups; g++) {
      int base = g * channels;
      for (int c = 0; c < channels; c++) {
        sums[c] += partials[base + c];
      }
    }
  }

  /** Single-precision partial buffer, as the Vulkan pipeline writes it. */
  protected final void accumulate(float[] partials, int groups, double[] sums) {
    for (int g = 0; g < groups; g++) {
      int base = g * channels;
      for (int c = 0; c < channels; c++) {
        sums[c] += partials[base + c];
      }
    }
  }

  /**
   * The summed channels divided by the scenario count. The device kernels do
   * not carry a sum of squares, so the standard error is left unestimated.
   */
  protected final AadResult finish(double[] sums, long paths, double seconds) {
    double[] gradients = new double[tape.inputCount()];
    for (int j = 0; j + 1 < channels; j++) {
      gradients[j] = sums[j + 1] / paths;
    }
    return new AadResult(sums[0] / paths, gradients, tape.inputNames(), paths, seconds);
  }

  /**
   * Per-dispatch wall-clock budget, honouring
   * {@code -Dnablatensor.maxLaunchSeconds}. A GPU is the one backend where an
   * overlong dispatch is actively dangerous: when the device also drives a
   * display, the driver's watchdog recovers by resetting it, so each engine
   * passes a {@code fallback} an order of magnitude below its own watchdog.
   */
  public static double maxLaunchSeconds(double fallback) {
    String override = System.getProperty("nablatensor.maxLaunchSeconds");
    return override != null ? Double.parseDouble(override) : fallback;
  }
}
