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
package com.nablatensor.backend.cuda;

/**
 * Public, minimal handle on the CUDA driver for code that generates its own
 * kernel source at runtime (NVRTC compile, device allocate, launch). The
 * tensor/jit path in {@link CudaBackend} keeps its own fused-kernel cache; this
 * facade exists so other modules can drive the same driver context and stream
 * without a second CUDA context being created.
 */
public final class CudaJit {

  private CudaJit() {
  }

  public static boolean isAvailable() {
    try {
      return CudaRuntime.probe();
    } catch (Throwable failure) {
      return false;
    }
  }

  public static String deviceName() {
    return CudaRuntime.context().name();
  }

  /** NVRTC target architecture of device 0, e.g. {@code compute_75}. */
  public static String architecture() {
    return CudaRuntime.context().arch();
  }

  /**
   * Whether the driver will kill kernels that run too long on this device
   * ({@code CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT}). This is set precisely
   * when the device is also driving a display, in which case an overlong launch
   * does not merely fail: it stalls the compositor for the duration and, once
   * the watchdog fires, takes the screen down with it through a GPU reset.
   * Callers should bound their launch durations when this is true.
   */
  public static boolean kernelTimeoutEnabled() {
    return CudaRuntime.deviceAttribute(KERNEL_EXEC_TIMEOUT) != 0;
  }

  private static final int KERNEL_EXEC_TIMEOUT = 17;

  /** Compiles CUDA C source with NVRTC and returns a launchable function handle. */
  public static long compile(String source, String kernelName) {
    CudaRuntime.DeviceInfo info = CudaRuntime.context();
    return CudaRuntime.loadFunction(CudaRuntime.compile(source, info.arch()), kernelName);
  }

  public static long malloc(long bytes) {
    CudaRuntime.context();
    return CudaRuntime.malloc(bytes);
  }

  public static void free(long pointer) {
    CudaRuntime.free(pointer);
  }

  public static void uploadDoubles(long pointer, double[] data) {
    CudaRuntime.uploadDoubles(pointer, data);
  }

  /** Uploads a tape's structural arrays (opcodes, argument indices, flags). */
  public static void uploadInts(long pointer, int[] data) {
    CudaRuntime.uploadIntsAsync(pointer, data);
  }

  public static double[] downloadDoubles(long pointer, int count) {
    return CudaRuntime.downloadDoubles(pointer, count);
  }

  /** Launches a 1-D grid. Arguments must be {@code Long}, {@code Integer} or {@code Float}. */
  public static void launch(long function, int grid, int block, Object... arguments) {
    CudaRuntime.launch(function, grid, block, arguments);
  }

  public static void synchronize() {
    CudaRuntime.syncStream();
  }
}
