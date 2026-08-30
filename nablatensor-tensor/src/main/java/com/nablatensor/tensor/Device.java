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
package com.nablatensor.tensor;

/** A concrete compute device: a {@link DeviceType} plus an ordinal index. */
public record Device(DeviceType type, int index) {

  public static Device cpu() {
    return new Device(DeviceType.CPU, 0);
  }

  public static Device cuda() {
    return new Device(DeviceType.CUDA, 0);
  }

  public static Device cuda(int index) {
    return new Device(DeviceType.CUDA, index);
  }

  public static Device rocm() {
    return new Device(DeviceType.ROCM, 0);
  }

  public static Device rocm(int index) {
    return new Device(DeviceType.ROCM, index);
  }

  public static Device vulkan() {
    return new Device(DeviceType.VULKAN, 0);
  }

  public static Device vulkan(int index) {
    return new Device(DeviceType.VULKAN, index);
  }

  @Override
  public String toString() {
    return type.name().toLowerCase() + ":" + index;
  }
}
