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

/** {@link DeviceRuntime} over the CUDA/NVRTC facade. */
final class CudaDeviceRuntime implements DeviceRuntime {

  static final CudaDeviceRuntime INSTANCE = new CudaDeviceRuntime();

  private CudaDeviceRuntime() {
  }

  @Override
  public long compile(String source, String kernelName) {
    return CudaJit.compile(source, kernelName);
  }

  @Override
  public long malloc(long bytes) {
    return CudaJit.malloc(bytes);
  }

  @Override
  public void free(long pointer) {
    CudaJit.free(pointer);
  }

  @Override
  public void uploadDoubles(long pointer, double[] data) {
    CudaJit.uploadDoubles(pointer, data);
  }

  @Override
  public double[] downloadDoubles(long pointer, int count) {
    return CudaJit.downloadDoubles(pointer, count);
  }

  @Override
  public void launch(long kernel, int groups, int local, Object... args) {
    CudaJit.launch(kernel, groups, local, args);
  }

  @Override
  public void synchronize() {
    CudaJit.synchronize();
  }
}
