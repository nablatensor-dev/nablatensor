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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.nablatensor.tensor.Op;
import com.nablatensor.tensor.Shape;
import org.junit.jupiter.api.Test;

/**
 * The host-side launch math the CUDA, ROCm, and Vulkan backends share. These
 * values are wired straight into kernel arguments, so a change here that the
 * kernels do not expect would break every GPU backend at once.
 */
class BackendSupportTest {

  @Test
  void grid1dRoundsUpAndZeroStaysZero() {
    assertEquals(0, GpuLaunch.grid1d(0));
    assertEquals(1, GpuLaunch.grid1d(1));
    assertEquals(1, GpuLaunch.grid1d(256));
    assertEquals(2, GpuLaunch.grid1d(257));
    assertEquals(4, GpuLaunch.grid1d(1024, 256));
    assertEquals(3, GpuLaunch.ceilDiv(129, 64));
  }

  @Test
  void opCodesMatchTheKernelSwitchLabels() {
    assertEquals(0, OpCodes.binary(Op.ADD));
    assertEquals(3, OpCodes.binary(Op.DIV));
    assertEquals(5, OpCodes.binary(Op.MIN));
    assertEquals(1, OpCodes.unary(Op.EXP));
    assertEquals(6, OpCodes.unary(Op.SIGMOID));
    assertEquals(9, OpCodes.unary(Op.SIGN));
    assertThrows(IllegalArgumentException.class, () -> OpCodes.binary(Op.EXP));
    assertThrows(IllegalArgumentException.class, () -> OpCodes.unary(Op.ADD));
  }

  @Test
  void axisReductionDecomposesAndReshapes() {
    AxisReduction r = AxisReduction.of(Shape.of(2, 5, 3, 4), 1);
    assertEquals(2, r.outer());
    assertEquals(5, r.axisSize());
    assertEquals(12, r.inner());
    assertEquals(24, r.outputSize());
    assertArrayEquals(new int[] {2, 3, 4}, r.outputShape(false).dims());
    assertArrayEquals(new int[] {2, 1, 3, 4}, r.outputShape(true).dims());

    AxisReduction last = AxisReduction.of(Shape.of(6, 7), 1);
    assertEquals(6, last.outer());
    assertEquals(7, last.axisSize());
    assertEquals(1, last.inner());
    assertArrayEquals(new int[] {6}, last.outputShape(false).dims());

    assertThrows(IllegalArgumentException.class, () -> AxisReduction.of(Shape.of(3, 3), 2));
    assertThrows(IllegalArgumentException.class, () -> AxisReduction.of(Shape.of(3, 3), -1));
  }

  @Test
  void axisReductionGradientCheck() {
    AxisReduction r = AxisReduction.of(Shape.of(4, 8), 1);
    r.requireGradient(Shape.of(4), 4);
    assertThrows(IllegalArgumentException.class, () -> r.requireGradient(Shape.of(8), 8));
  }

  @Test
  void broadcastLayoutPadsToRank4WithStrideZeroOnBroadcastAxes() {
    // (1,4,1) -> (3,4,5): axis 0 and 2 are broadcast (stride 0), axis 1 keeps stride 1
    BroadcastLayout layout = BroadcastLayout.of(Shape.of(1, 4, 1), Shape.of(3, 4, 5));
    assertArrayEquals(new int[] {1, 3, 4, 5}, layout.dims());
    assertArrayEquals(new int[] {0, 0, 1, 0}, layout.strides());

    // a full-rank exact match: contiguous row-major strides, nothing broadcast
    BroadcastLayout exact = BroadcastLayout.of(Shape.of(2, 3), Shape.of(2, 3));
    assertArrayEquals(new int[] {1, 1, 2, 3}, exact.dims());
    assertArrayEquals(new int[] {0, 0, 3, 1}, exact.strides());

    assertThrows(UnsupportedOperationException.class,
        () -> BroadcastLayout.of(Shape.of(1), Shape.of(2, 2, 2, 2, 2)));
  }
}
