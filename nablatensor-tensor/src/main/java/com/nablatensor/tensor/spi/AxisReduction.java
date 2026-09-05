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
package com.nablatensor.tensor.spi;

import com.nablatensor.tensor.Shape;

/**
 * The {@code outer × axisSize × inner} view of a tensor reduced along one axis,
 * plus the result shape. Every backend's {@code sumAxis} / {@code maxAxis} /
 * {@code argMaxAxis} / {@code maxAxisBackward} builds this same decomposition and
 * feeds {@code outer}, {@code axisSize}, {@code inner} to its {@code
 * reduce_axis_*} kernel; computing it here keeps the index math and the bounds
 * check identical across CUDA, ROCm, and Vulkan.
 */
public record AxisReduction(Shape shape, int axis, int outer, int axisSize, int inner) {

  /** Decomposes {@code shape} about {@code axis}, validating the axis is in range. */
  public static AxisReduction of(Shape shape, int axis) {
    if (axis < 0 || axis >= shape.rank()) {
      throw new IllegalArgumentException("axis " + axis + " is out of bounds for shape " + shape);
    }
    int outer = 1;
    for (int i = 0; i < axis; i++) {
      outer = Math.multiplyExact(outer, shape.dim(i));
    }
    int inner = 1;
    for (int i = axis + 1; i < shape.rank(); i++) {
      inner = Math.multiplyExact(inner, shape.dim(i));
    }
    return new AxisReduction(shape, axis, outer, shape.dim(axis), inner);
  }

  /** Output element count: one per {@code (outer, inner)} pair. */
  public int outputSize() {
    return Math.multiplyExact(outer, inner);
  }

  /** Result shape: {@code axis} kept as size 1 when {@code keepDims}, otherwise dropped. */
  public Shape outputShape(boolean keepDims) {
    int[] dims = shape.dims();
    if (keepDims) {
      dims[axis] = 1;
      return Shape.of(dims);
    }
    int[] reduced = new int[dims.length - 1];
    System.arraycopy(dims, 0, reduced, 0, axis);
    System.arraycopy(dims, axis + 1, reduced, axis, dims.length - axis - 1);
    return Shape.of(reduced);
  }

  /**
   * Validates that an upstream-gradient buffer carries exactly one value per
   * reduced slice, as {@code maxAxisBackward} requires.
   */
  public void requireGradient(Shape gradientShape, int gradientCount) {
    if (gradientCount != outputSize()) {
      throw new IllegalArgumentException("maxAxisBackward gradient shape " + gradientShape
          + " is incompatible with input " + shape + " and axis " + axis);
    }
  }
}
