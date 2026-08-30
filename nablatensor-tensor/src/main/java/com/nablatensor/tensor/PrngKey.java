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

import java.util.ArrayList;
import java.util.List;

/**
 * Immutable counter-based random key. Reusing a key reproduces the same values;
 * use {@link #advance(long)}, {@link #split(int)}, or {@link #foldIn(long)} to
 * derive independent deterministic streams.
 */
public record PrngKey(long seed, long counter) {

  public static PrngKey of(long seed) {
    return new PrngKey(seed, 0);
  }

  public PrngKey advance(long amount) {
    if (amount < 0) {
      throw new IllegalArgumentException("PRNG counter advance must be non-negative");
    }
    return new PrngKey(seed, Math.addExact(counter, amount));
  }

  public PrngKey foldIn(long data) {
    return new PrngKey(mix64(seed ^ mix64(data)), counter);
  }

  public List<PrngKey> split(int count) {
    if (count < 1) {
      throw new IllegalArgumentException("PRNG split count must be positive");
    }
    List<PrngKey> keys = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      keys.add(new PrngKey(mix64(seed + 0x9E3779B97F4A7C15L * (counter + i + 1)), 0));
    }
    return List.copyOf(keys);
  }

  public Tensor uniform(Device device, int... dims) {
    Shape shape = checkedShape(dims);
    ComputeBackend backend = BackendRegistry.forDevice(device);
    return new Tensor(backend, backend.randomUniform(seed, counter, shape, device));
  }

  public Tensor normal(Device device, int... dims) {
    Shape shape = checkedShape(dims);
    ComputeBackend backend = BackendRegistry.forDevice(device);
    return new Tensor(backend, backend.randomNormal(seed, counter, shape, device));
  }

  public Tensor uniform(int... dims) {
    return uniform(NablaTensors.defaultDevice(), dims);
  }

  public Tensor normal(int... dims) {
    return normal(NablaTensors.defaultDevice(), dims);
  }

  private static Shape checkedShape(int[] dims) {
    if (dims.length == 0) {
      throw new IllegalArgumentException("random tensor shape must have at least one dimension");
    }
    long size = 1;
    for (int dim : dims) {
      if (dim < 1) {
        throw new IllegalArgumentException("random tensor dimensions must be positive");
      }
      size = Math.multiplyExact(size, dim);
    }
    Math.toIntExact(size);
    return Shape.of(dims);
  }

  private static long mix64(long value) {
    long mixed = value;
    mixed = (mixed ^ (mixed >>> 30)) * 0xBF58476D1CE4E5B9L;
    mixed = (mixed ^ (mixed >>> 27)) * 0x94D049BB133111EBL;
    return mixed ^ (mixed >>> 31);
  }
}
