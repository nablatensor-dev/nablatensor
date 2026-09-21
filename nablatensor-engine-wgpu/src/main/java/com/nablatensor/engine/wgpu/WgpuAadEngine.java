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
package com.nablatensor.engine.wgpu;

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadExecutable;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadTape;

/**
 * wgpu-native replay engine. Priority sits just below Vulkan: on this machine
 * both ultimately drive the same RADV Vulkan ICD, so Vulkan's lower per-call
 * overhead (persistent host-mapped buffers, no async adapter/device
 * handshake) wins the default selection; wgpu's payoff is portability — the
 * same engine also reaches Metal and D3D12 through wgpu-native's other
 * backends, which Vulkan cannot.
 *
 * <p>Contains no reference to {@code libwgpu_native} beyond
 * {@link #isAvailable()}, which probes the runtime through a facade, so
 * merely enumerating engines cannot fail where wgpu-native is absent.
 */
public final class WgpuAadEngine implements AadEngine {

  @Override
  public String name() {
    return "wgpu";
  }

  @Override
  public int priority() {
    return 58;
  }

  @Override
  public boolean isAvailable() {
    return WgpuAadKernel.wgpuAvailable();
  }

  @Override
  public boolean supports(AadOptions options) {
    return options.precision() == AadOptions.PrecisionEnum.FLOAT32;
  }

  @Override
  public String describe() {
    if (!isAvailable()) {
      return "wgpu (WebGPU) unavailable";
    }
    return "wgpu · " + WgpuAadKernel.wgpuDeviceName() + " · WGSL compute · fp32";
  }

  @Override
  public AadExecutable compile(AadTape tape, AadOptions options) {
    requireSupportedTape(tape);
    return WgpuAadKernel.compile(tape, options);
  }
}
