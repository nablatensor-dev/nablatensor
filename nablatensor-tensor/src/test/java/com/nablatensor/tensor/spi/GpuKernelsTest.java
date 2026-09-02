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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertLinesMatch;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

/**
 * Pins the kernel catalog. The names are parsed out of the CUDA-C source, so
 * without this test a kernel that stops being recognised would silently vanish
 * from the module instead of failing the build.
 */
class GpuKernelsTest {

  private static final List<String> EXPECTED_NAMES = List.of(
      "ew_binary", "ew_scalar", "ew_unary", "relu_backward",
      "transpose2d", "matmul_tiled", "batched_matmul_tiled",
      "random_uniform", "random_normal",
      "reduce_sum", "reduce_max", "sum_axis0", "reduce_axis_sum", "reduce_axis_max",
      "reduce_axis_argmax", "reduce_axis_max_backward", "broadcast_to",
      "conv2d_fwd", "conv2d_dx", "conv2d_dw");

  @Test
  void catalogMatchesTheKernelsTheBackendsLoad() {
    assertLinesMatch(EXPECTED_NAMES, Arrays.asList(GpuKernels.TENSOR_KERNEL_NAMES));
  }

  @Test
  void everyNamedKernelIsDefinedInTheTranslationUnit() {
    for (String name : GpuKernels.TENSOR_KERNEL_NAMES) {
      assertTrue(GpuKernels.TENSOR_SOURCE.contains("__global__ void " + name),
          name + " is not defined in TENSOR_SOURCE");
    }
  }

  @Test
  void deviceHelpersArePrependedBeforeTheirCallers() {
    assertTrue(GpuKernels.TENSOR_SOURCE.indexOf("__device__ __forceinline__ float random_uniform_value")
        < GpuKernels.TENSOR_SOURCE.indexOf("__global__ void random_uniform("));
  }

  @Test
  void signaturesAreReadFromTheSource() {
    GpuKernel matmul = GpuKernels.kernel("matmul_tiled");
    assertEquals(7, matmul.arity());
    assertEquals("unsigned long long", matmul.paramTypes().get(6));
    assertEquals(16, matmul.blockDimX());
    assertEquals(16, matmul.blockDimY());
  }

  @Test
  void sourceWithoutAnEntryPointIsRejected() {
    assertThrows(IllegalArgumentException.class,
        () -> GpuKernel.of("__device__ float half(float x) { return 0.5f * x; }\n"));
  }

  @Test
  void sourceWithTwoEntryPointsIsRejected() {
    assertThrows(IllegalArgumentException.class, () -> GpuKernel.of("""
        extern "C" __global__ void a(float* out) { out[0] = 0.0f; }
        extern "C" __global__ void b(float* out) { out[0] = 1.0f; }
        """));
  }
}
