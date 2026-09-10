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

/**
 * CUDA replay engine. Highest priority, so a machine with a device always uses
 * it unless another engine is named explicitly.
 */
public final class CudaAadEngine implements AadEngine {

  @Override
  public String name() {
    return "cuda";
  }

  @Override
  public int priority() {
    return 100;
  }

  @Override
  public boolean isAvailable() {
    try {
      return CudaJit.isAvailable();
    } catch (Throwable ignored) {
      return false;
    }
  }

  @Override
  public boolean supports(AadOptions options) {
    return true;
  }

  @Override
  public String describe() {
    try {
      return "CUDA · " + CudaJit.deviceName() + " · " + CudaJit.architecture();
    } catch (Throwable ignored) {
      return "CUDA · no device";
    }
  }

  @Override
  public AadExecutable compile(AadTape tape, AadOptions options) {
    requireSupportedTape(tape);
    if (!CudaJit.isAvailable()) {
      throw new IllegalStateException("no CUDA device available for the AAD replay kernel");
    }
    return DeviceAadExecutable.compile(tape, options, "cuda", CudaDeviceRuntime.INSTANCE,
        DeviceAadExecutable.CUDA_C, maxLaunchSeconds());
  }

  /**
   * A device that also drives a display has the driver's
   * {@code KERNEL_EXEC_TIMEOUT} watchdog behind it, and that watchdog recovers
   * by resetting the GPU, so the budget there is an order of magnitude below it.
   */
  private static double maxLaunchSeconds() {
    return GpuAadExecutable.maxLaunchSeconds(CudaJit.kernelTimeoutEnabled() ? 0.2 : 2.0);
  }
}
