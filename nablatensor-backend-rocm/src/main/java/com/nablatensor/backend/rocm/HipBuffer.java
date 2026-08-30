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

import com.nablatensor.tensor.DType;
import com.nablatensor.tensor.Device;
import com.nablatensor.tensor.Shape;
import com.nablatensor.tensor.spi.DeviceBuffer;

import java.lang.ref.Cleaner;

/**
 * A HIP device allocation identified by its driver pointer address. Tensors are
 * immutable and fluent chains drop intermediates immediately, so buffers are
 * normally freed via a {@link Cleaner} once the wrapper is unreachable;
 * {@link #release()} lets a caller free deterministically instead.
 */
final class HipBuffer implements DeviceBuffer {

  private static final Cleaner CLEANER = Cleaner.create();

  private record FreeAction(long pointer) implements Runnable {
    @Override
    public void run() {
      HipRuntime.enqueueFree(pointer);
    }
  }

  final long pointer;
  private final Shape shape;
  private final DType dtype;
  private final Device device;
  private final Cleaner.Cleanable cleanable;

  HipBuffer(long pointer, Shape shape, DType dtype, Device device) {
    this.pointer = pointer;
    this.shape = shape;
    this.dtype = dtype;
    this.device = device;
    this.cleanable = CLEANER.register(this, new FreeAction(pointer));
  }

  void release() {
    if (cleanable != null) {
      cleanable.clean();
      HipRuntime.drainPendingFrees();
    }
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
