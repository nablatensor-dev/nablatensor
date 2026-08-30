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
package com.nablatensor.backend.vulkan;

import com.nablatensor.tensor.DType;
import com.nablatensor.tensor.Device;
import com.nablatensor.tensor.Shape;
import com.nablatensor.tensor.spi.DeviceBuffer;

import java.lang.ref.Cleaner;

/**
 * A Vulkan storage buffer plus its backing device memory. Freed via a
 * {@link Cleaner} once unreachable, or eagerly with {@link #release()}.
 */
final class VulkanBuffer implements DeviceBuffer {

  private static final Cleaner CLEANER = Cleaner.create();

  private record FreeAction(long buffer, long memory) implements Runnable {
    @Override
    public void run() {
      try {
        VulkanRuntime.free(buffer, memory);
      } catch (RuntimeException ignored) {
        // teardown races are not fatal
      }
    }
  }

  final long buffer;
  final long memory;
  private final Shape shape;
  private final DType dtype;
  private final Device device;
  private final Cleaner.Cleanable cleanable;

  VulkanBuffer(long buffer, long memory, Shape shape, DType dtype, Device device) {
    this.buffer = buffer;
    this.memory = memory;
    this.shape = shape;
    this.dtype = dtype;
    this.device = device;
    this.cleanable = CLEANER.register(this, new FreeAction(buffer, memory));
  }

  void release() {
    cleanable.clean();
  }

  int count() {
    return Math.toIntExact(shape.size());
  }

  @Override
  public Shape shape() {
    return shape;
  }

  @Override
  public DType dtype() {
    return dtype;
  }

  @Override
  public Device device() {
    return device;
  }
}
