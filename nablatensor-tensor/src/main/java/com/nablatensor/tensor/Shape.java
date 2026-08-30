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

import java.util.Arrays;

/** An immutable tensor shape. */
public record Shape(int[] dims) {

  public Shape {
    dims = dims.clone();
  }

  public static Shape of(int... dims) {
    return new Shape(dims);
  }

  public int rank() {
    return dims.length;
  }

  public int dim(int axis) {
    return dims[axis];
  }

  public long size() {
    long total = 1;
    for (int d : dims) {
      total = Math.multiplyExact(total, d);
    }
    return total;
  }

  /**
   * Returns a shape with the same element count. One dimension may be {@code -1}
   * and is inferred from the remaining dimensions.
   */
  public Shape reshape(int... requestedDims) {
    int[] resolved = requestedDims.clone();
    int inferredAxis = -1;
    long knownSize = 1;
    for (int i = 0; i < resolved.length; i++) {
      int dim = resolved[i];
      if (dim == -1) {
        if (inferredAxis >= 0) {
          throw new IllegalArgumentException(
              "reshape permits one inferred dimension: " + this + " -> " + Arrays.toString(requestedDims));
        }
        inferredAxis = i;
      } else if (dim <= 0) {
        throw new IllegalArgumentException(
            "reshape dimensions must be positive or -1: " + this + " -> " + Arrays.toString(requestedDims));
      } else {
        knownSize = Math.multiplyExact(knownSize, dim);
      }
    }

    long sourceSize = size();
    if (inferredAxis >= 0) {
      if (knownSize == 0 || sourceSize % knownSize != 0) {
        throw new IllegalArgumentException(
            "reshape element count mismatch: " + this + " -> " + Arrays.toString(requestedDims));
      }
      resolved[inferredAxis] = Math.toIntExact(sourceSize / knownSize);
    } else if (knownSize != sourceSize) {
      throw new IllegalArgumentException(
          "reshape element count mismatch: " + this + " -> " + Arrays.toString(requestedDims));
    }
    return Shape.of(resolved);
  }

  /** 2-D transpose; only defined for rank-2 shapes. */
  public Shape transposed() {
    if (dims.length != 2) {
      throw new IllegalStateException("transpose requires a rank-2 shape, got " + this);
    }
    return Shape.of(dims[1], dims[0]);
  }

  /**
   * NumPy-style broadcast result shape: shapes are right-aligned, missing
   * leading dims count as 1, and each aligned pair must be equal or one of
   * them must be 1.
   */
  public static Shape broadcast(Shape a, Shape b) {
    int rank = Math.max(a.rank(), b.rank());
    int[] out = new int[rank];
    for (int i = 0; i < rank; i++) {
      int da = dimOrOne(a, rank, i);
      int db = dimOrOne(b, rank, i);
      if (da != db && da != 1 && db != 1) {
        throw new IllegalArgumentException("cannot broadcast shapes " + a + " and " + b);
      }
      out[i] = Math.max(da, db);
    }
    return Shape.of(out);
  }

  private static int dimOrOne(Shape s, int rank, int i) {
    int pad = rank - s.rank();
    return i < pad ? 1 : s.dim(i - pad);
  }

  @Override
  public int[] dims() {
    return dims.clone();
  }

  @Override
  public boolean equals(Object o) {
    return o instanceof Shape s && Arrays.equals(dims, s.dims);
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(dims);
  }

  @Override
  public String toString() {
    return Arrays.toString(dims);
  }
}
