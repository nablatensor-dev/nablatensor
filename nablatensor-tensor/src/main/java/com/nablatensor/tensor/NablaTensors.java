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

import com.nablatensor.tensor.spi.ComputeBackend;
import com.nablatensor.tensor.spi.DeviceBuffer;

import java.util.List;
import java.util.Random;

/** Fluent entry point for creating tensors and querying devices/backends. */
public final class NablaTensors {

  private NablaTensors() {
  }

  // ---- device / backend introspection ------------------------------------

  public static List<ComputeBackend> backends() {
    return BackendRegistry.available();
  }

  /** Devices available on this machine, one per available backend. */
  public static List<Device> devices() {
    return BackendRegistry.available().stream()
        .map(b -> new Device(b.deviceType(), 0))
        .toList();
  }

  public static Device defaultDevice() {
    ComputeBackend backend = BackendRegistry.defaultBackend();
    return new Device(backend.deviceType(), 0);
  }

  /** Creates an immutable reproducible counter-based random key. */
  public static PrngKey key(long seed) {
    return PrngKey.of(seed);
  }

  // ---- tensor construction ------------------------------------------------

  public static Tensor array(float[] data, int... dims) {
    Shape shape = Shape.of(dims.length == 0 ? new int[] {data.length} : dims);
    return arrayOn(data, shape, defaultDevice());
  }

  public static Tensor array(float[][] data) {
    int rows = data.length;
    int cols = data[0].length;
    float[] flat = new float[rows * cols];
    for (int r = 0; r < rows; r++) {
      System.arraycopy(data[r], 0, flat, r * cols, cols);
    }
    return arrayOn(flat, Shape.of(rows, cols), defaultDevice());
  }

  public static Tensor arrayOn(float[] data, Shape shape, Device device) {
    ComputeBackend backend = BackendRegistry.forDevice(device);
    DeviceBuffer buffer = backend.upload(data, shape, DType.F32, device);
    return new Tensor(backend, buffer);
  }

  public static Tensor zeros(int... dims) {
    Shape shape = Shape.of(dims);
    return arrayOn(new float[(int) shape.size()], shape, defaultDevice());
  }

  public static Tensor randn(int... dims) {
    return randn(new Random(), dims);
  }

  public static Tensor randn(Random random, int... dims) {
    Shape shape = Shape.of(dims);
    float[] data = new float[(int) shape.size()];
    for (int i = 0; i < data.length; i++) {
      data[i] = (float) random.nextGaussian();
    }
    return arrayOn(data, shape, defaultDevice());
  }
}
