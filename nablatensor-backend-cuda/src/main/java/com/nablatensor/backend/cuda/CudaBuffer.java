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
package com.nablatensor.backend.cuda;

import com.nablatensor.tensor.DType;
import com.nablatensor.tensor.Device;
import com.nablatensor.tensor.Shape;
import com.nablatensor.tensor.spi.DeviceBuffer;

import java.lang.ref.Cleaner;

/**
 * A CUDA device allocation identified by its driver pointer address.
 *
 * <p>Tensors are immutable and fluent chains discard intermediates immediately,
 * so buffers are normally freed via a {@link Cleaner} once the wrapper is
 * unreachable. {@link #release()} additionally lets a caller (the {@code Jit}
 * tracer, for its hard-break intermediates) free deterministically instead of
 * waiting on GC timing; {@link Cleaner.Cleanable#clean()} guarantees the
 * underlying free only ever runs once, so this is safe even if the Cleaner
 * later processes the same (by-then-unreachable) object too. {@link #LIVE_BYTES}
 * tracks how much device memory is currently allocated (including buffers only
 * reachable via the Cleaner, not yet freed) so {@code CudaRuntime.malloc} can
 * enforce a memory budget rather than only reacting once allocation fails.
 */
final class CudaBuffer implements DeviceBuffer {

  private static final Cleaner CLEANER = Cleaner.create();
  static final java.util.concurrent.atomic.AtomicLong LIVE_BYTES = new java.util.concurrent.atomic.AtomicLong();

  private record FreeAction(long pointer, long bytes) implements Runnable {
    @Override
    public void run() {
      CudaRuntime.enqueueFree(pointer, bytes);
    }
  }

  final long pointer;
  private final Shape shape;
  private final DType dtype;
  private final Device device;
  @SuppressWarnings("unused")
  private final CudaBuffer storage;
  private final Cleaner.Cleanable cleanable;

  CudaBuffer(long pointer, Shape shape, DType dtype, Device device) {
    this.pointer = pointer;
    this.shape = shape;
    this.dtype = dtype;
    this.device = device;
    this.storage = null;
    long bytes = shape.size() * dtype.byteSize();
    LIVE_BYTES.addAndGet(bytes);
    this.cleanable = CLEANER.register(this, new FreeAction(pointer, bytes));
  }

  CudaBuffer(CudaBuffer storage, int elementOffset, int count) {
    if (elementOffset < 0 || count < 0 || (long) elementOffset + count > storage.shape.size()) {
      throw new IllegalArgumentException("CUDA view exceeds storage bounds");
    }
    this.pointer = storage.pointer + (long) elementOffset * storage.dtype.byteSize();
    this.shape = Shape.of(count);
    this.dtype = storage.dtype;
    this.device = storage.device;
    this.storage = storage;
    this.cleanable = null;
  }

  void release() {
    if (cleanable != null) {
      cleanable.clean();
      CudaRuntime.drainPendingFrees();
    }
  }

  int count() {
    return (int) shape.size();
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
