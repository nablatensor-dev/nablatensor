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
package com.nablatensor.backend.opencl;

/**
 * Public facade over the package-private {@link OpenClRuntime}, for code outside
 * this module that needs a raw OpenCL compute surface: build-from-source
 * compilation, device allocation, host&lt;-&gt;device double transfers and a 1-D
 * kernel launch.
 *
 * <p>The AAD OpenCL replay engine ({@code com.nablatensor.engine.opencl}) is the
 * intended caller; it mirrors {@code com.nablatensor.backend.rocm.HipCompute}.
 */
public final class OpenClCompute {

  private OpenClCompute() {
  }

  /** Whether a usable OpenCL device is present. Never throws. */
  public static boolean isAvailable() {
    try {
      return OpenClRuntime.probe();
    } catch (Throwable ignored) {
      return false;
    }
  }

  /**
   * Whether the selected device can compute in {@code double}. The replay kernel
   * keeps its accumulators and the host-visible partials in {@code double} even
   * when the working precision is fp32, so a device without {@code cl_khr_fp64}
   * cannot run it and the engine must decline.
   */
  public static boolean supportsFp64() {
    try {
      return OpenClRuntime.context().fp64();
    } catch (Throwable ignored) {
      return false;
    }
  }

  /** Largest work-group the device accepts; the replay kernel needs at least 256. */
  public static int maxWorkGroupSize() {
    try {
      return OpenClRuntime.context().maxWorkGroupSize();
    } catch (Throwable ignored) {
      return 0;
    }
  }

  /** {@code "<platform> · <device>"} for diagnostics, or a placeholder. */
  public static String describe() {
    try {
      OpenClRuntime.DeviceInfo info = OpenClRuntime.context();
      return info.platformName() + " · " + info.name() + " · " + info.version();
    } catch (Throwable ignored) {
      return "no device";
    }
  }

  public static String deviceName() {
    try {
      return OpenClRuntime.context().name();
    } catch (Throwable ignored) {
      return "no device";
    }
  }

  /**
   * Builds OpenCL C {@code source} and returns a launchable kernel handle for
   * {@code kernelName}. Callers should cache the handle by source; this does not.
   */
  public static long compile(String source, String kernelName) {
    return compile(source, kernelName, System.getProperty("nablatensor.opencl.buildOptions", ""));
  }

  /** As {@link #compile(String, String)} but with explicit {@code clBuildProgram} flags. */
  public static long compile(String source, String kernelName, String buildOptions) {
    return OpenClRuntime.compile(source, kernelName, buildOptions);
  }

  public static void releaseKernel(long kernel) {
    OpenClRuntime.releaseKernel(kernel);
  }

  public static long malloc(long bytes) {
    return OpenClRuntime.malloc(bytes);
  }

  public static void free(long buffer) {
    OpenClRuntime.free(buffer);
  }

  /** Blocking host-to-device copy into an already-allocated {@code buffer}. */
  public static void uploadDoubles(long buffer, double[] data) {
    OpenClRuntime.uploadDoubles(buffer, data);
  }

  public static double[] downloadDoubles(long buffer, int count) {
    return OpenClRuntime.downloadDoubles(buffer, count);
  }

  /**
   * Launches a 1-D range of {@code groups} work-groups of {@code local}
   * work-items. Arguments are {@link Long} (device buffer handles / 64-bit
   * scalars), {@link Integer} or {@link Float}, matched positionally.
   */
  public static void launch(long kernel, int groups, int local, Object... arguments) {
    OpenClRuntime.launch(kernel, groups, local, arguments);
  }

  public static void synchronize() {
    OpenClRuntime.synchronize();
  }
}
