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
package com.nablatensor.engine.vulkan;

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadExecutable;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadTape;

/**
 * Vulkan replay engine. Priority sits above SIMD and below CUDA, so on a
 * machine with no CUDA device but a working Vulkan compute stack it is the
 * engine {@code Nabla.model(...).fastest()} selects.
 *
 * <p>Contains no reference to {@code libvulkan} beyond {@link #isAvailable()},
 * which probes the runtime through a facade, so merely enumerating engines
 * cannot fail where Vulkan is absent.
 */
public final class VulkanAadEngine implements AadEngine {

  @Override
  public String name() {
    return "vulkan";
  }

  @Override
  public int priority() {
    return 60;
  }

  @Override
  public boolean isAvailable() {
    return VulkanAadKernel.vulkanAvailable();
  }

  @Override
  public boolean supports(AadOptions options) {
    return options.precision() == AadOptions.Precision.FLOAT32;
  }

  @Override
  public String describe() {
    if (!isAvailable()) {
      return "Vulkan compute unavailable";
    }
    return "Vulkan · " + VulkanAadKernel.vulkanDeviceName() + " · SPIR-V compute · fp32";
  }

  @Override
  public AadExecutable compile(AadTape tape, AadOptions options) {
    AadEngine.requireBasicRandom(tape, "vulkan");
    AadEngine.requireSingleOutput(tape, "vulkan");
    return VulkanAadKernel.compile(tape, options);
  }
}
