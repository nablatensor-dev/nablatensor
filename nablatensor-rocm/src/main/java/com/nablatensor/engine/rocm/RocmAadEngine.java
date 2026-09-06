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
package com.nablatensor.engine.rocm;

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadExecutable;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadTape;
import com.nablatensor.backend.rocm.HipCompute;

/**
 * ROCm/HIP replay engine. Priority sits above SIMD and below the Vulkan engine.
 * {@link #isAvailable()} returns {@code false} on the known-unstable consumer
 * APUs ({@code gfx1103} and friends), where a sustained-compute launch wedges
 * the MES firmware and forces a display-killing full-device reset, so
 * {@code .fastest()} never lands here on this dev box. Opt an APU back in for
 * deliberate benchmarking with {@code -Dnablatensor.rocm.allow_unsupported=true}
 * (env {@code NABLATENSOR_ROCM_ALLOW_UNSUPPORTED=1}) and then
 * {@code -Dnablatensor.engine=rocm} / {@code .on("rocm")}.
 *
 * <p>Holds no reference to {@code libamdhip64} beyond {@link #isAvailable()},
 * which probes through the {@link HipCompute} facade, so merely enumerating
 * engines cannot fail where ROCm is absent.
 */
public final class RocmAadEngine implements AadEngine {

  @Override
  public String name() {
    return "rocm";
  }

  @Override
  public int priority() {
    return 55;
  }

  @Override
  public boolean isAvailable() {
    try {
      return HipCompute.reliable();
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
      return "ROCm/HIP · " + HipCompute.deviceName() + " · " + HipCompute.arch() + " · HIPRTC";
    } catch (Throwable ignored) {
      return "ROCm/HIP · no device";
    }
  }

  @Override
  public AadExecutable compile(AadTape tape, AadOptions options) {
    AadEngine.requireBasicRandom(tape, "rocm");
    AadEngine.requireSingleOutput(tape, "rocm");
    return RocmAadKernel.compile(tape, options);
  }
}
