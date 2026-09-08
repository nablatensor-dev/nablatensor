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
package com.nablatensor.engine.opencl;

import com.nablatensor.backend.opencl.OpenClCompute;
import com.nablatensor.engine.DeviceRuntime;

/** {@link DeviceRuntime} over the OpenCL facade. */
final class OpenClDeviceRuntime implements DeviceRuntime {

  static final OpenClDeviceRuntime INSTANCE = new OpenClDeviceRuntime();

  private OpenClDeviceRuntime() {
  }

  /**
   * Flags handed to {@code clBuildProgram}. Only {@code -cl-mad-enable}
   * (multiply-add contraction — keeps IEEE round-to-nearest and NaN/Inf
   * semantics); the aggressive {@code -cl-fast-relaxed-math} was measured to add
   * nothing once the {@code native_*} transcendentals are in. Override with
   * {@code -Dnablatensor.opencl.buildOptions=...} (empty string for none).
   */
  static String buildOptions() {
    return System.getProperty("nablatensor.opencl.buildOptions", "-cl-mad-enable");
  }

  @Override
  public long compile(String source, String kernelName) {
    return OpenClCompute.compile(source, kernelName, buildOptions());
  }

  @Override
  public String compileKey() {
    return buildOptions();
  }

  @Override
  public long malloc(long bytes) {
    return OpenClCompute.malloc(bytes);
  }

  @Override
  public void free(long pointer) {
    OpenClCompute.free(pointer);
  }

  @Override
  public void uploadDoubles(long pointer, double[] data) {
    OpenClCompute.uploadDoubles(pointer, data);
  }

  @Override
  public double[] downloadDoubles(long pointer, int count) {
    return OpenClCompute.downloadDoubles(pointer, count);
  }

  @Override
  public void launch(long kernel, int groups, int local, Object... args) {
    OpenClCompute.launch(kernel, groups, local, args);
  }

  @Override
  public void synchronize() {
    // no-op: the blocking clEnqueueReadBuffer in downloadDoubles is the sync point.
  }
}
