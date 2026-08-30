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
package com.nablatensor.backend.rocm;

/**
 * Public facade over the package-private {@link HipRuntime}, for code outside
 * this module that needs a raw ROCm/HIP compute surface: HIPRTC compilation,
 * device allocation, host&lt;-&gt;device double transfers and a 1-D kernel launch.
 *
 * <p>The AAD ROCm replay engine ({@code com.nablatensor.engine.rocm}) is the intended
 * caller; it mirrors {@code com.nablatensor.backend.vulkan.VulkanCompute}.
 */
public final class HipCompute {

  private HipCompute() {
  }

  /** Whether a usable HIP device is present. Never throws. */
  public static boolean isAvailable() {
    try {
      return HipRuntime.probe();
    } catch (Throwable ignored) {
      return false;
    }
  }

  /** Device name for diagnostics, or a placeholder if the context is unavailable. */
  public static String deviceName() {
    try {
      return HipRuntime.context().name();
    } catch (Throwable ignored) {
      return "no device";
    }
  }

  /** The {@code --offload-arch} target HIPRTC will build for (e.g. {@code gfx1103}). */
  public static String arch() {
    return HipRuntime.context().arch();
  }

  /**
   * Compiles HIP/CUDA-C {@code source} with HIPRTC and returns a launchable
   * function handle for {@code kernelName}. Callers should cache the handle by
   * source; this does not.
   */
  public static long compile(String source, String kernelName) {
    return HipRuntime.loadFunction(HipRuntime.compile(source, HipRuntime.context().arch()), kernelName);
  }

  public static long malloc(long bytes) {
    return HipRuntime.malloc(bytes);
  }

  public static void free(long pointer) {
    HipRuntime.free(pointer);
  }

  /** Host-to-device copy into an already-allocated {@code pointer}. */
  public static void uploadDoubles(long pointer, double[] data) {
    HipRuntime.uploadDoubles(pointer, data);
  }

  public static double[] downloadDoubles(long pointer, int count) {
    return HipRuntime.downloadDoubles(pointer, count);
  }

  /**
   * Launches a 1-D grid of {@code grid} blocks of {@code block} threads.
   * Arguments are {@link Long} (device pointers / 64-bit scalars),
   * {@link Integer} or {@link Float}, matched positionally to the kernel.
   */
  public static void launch(long function, int grid, int block, Object... arguments) {
    HipRuntime.launch(function, grid, block, arguments);
  }

  public static void synchronize() {
    HipRuntime.synchronize();
  }
}
