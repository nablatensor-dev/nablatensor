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
 * The small accelerator surface {@link DeviceAadExecutable} needs: compile a
 * kernel from source, allocate and move linear device buffers, launch a 1-D
 * grid, and make prior work host-visible. Each accelerator backend supplies one
 * adapter over its own runtime facade (CUDA {@code CudaJit}, ROCm
 * {@code HipCompute}, OpenCL {@code OpenClCompute}); the replay bookkeeping that
 * used to be copied into every {@code *AadKernel} lives once in
 * {@link DeviceAadExecutable}.
 *
 * <p>Handles are opaque {@code long}s (device pointers / kernel functions) and
 * are only ever passed back to the same runtime instance.
 */
@Internal
public interface DeviceRuntime {

  /** Compile {@code source} and return a launchable handle for {@code kernelName}. */
  long compile(String source, String kernelName);

  long malloc(long bytes);

  void free(long pointer);

  /** Host-to-device copy into an already-allocated {@code pointer}. */
  void uploadDoubles(long pointer, double[] data);

  double[] downloadDoubles(long pointer, int count);

  /**
   * Launch a 1-D grid of {@code groups} work-groups of {@code local} work-items.
   * Arguments are matched positionally: {@link Long} (device buffer handle or
   * 64-bit scalar), {@link Integer} or {@link Float}.
   */
  void launch(long kernel, int groups, int local, Object... args);

  /**
   * Ensure the launched kernel has finished and its results are host-visible.
   * A runtime whose {@link #downloadDoubles} already blocks on prior work may
   * leave this empty.
   */
  void synchronize();

  /**
   * Extra text folded into the kernel cache key, for a runtime whose compiled
   * output depends on more than the source string (e.g. build options). Empty
   * for CUDA and ROCm.
   */
  default String compileKey() {
    return "";
  }
}
