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

import java.util.List;

/** One-thread-per-element kernels: no shared memory, no cross-thread traffic. */
final class ElementwiseKernels {

  private ElementwiseKernels() {
  }

  /** Two same-shape operands; {@code op} indexes the case labels, not {@link com.nablatensor.tensor.Op}. */
  static final GpuKernel EW_BINARY = GpuKernel.of("""
      extern "C" __global__ void ew_binary(float* out, const float* a, const float* b, int n, int op) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        float x = a[i], y = b[i], r = 0.0f;
        switch (op) {
          case 0: r = x + y; break;
          case 1: r = x - y; break;
          case 2: r = x * y; break;
          case 3: r = x / y; break;
          case 4: r = isnan(x) ? x : (isnan(y) ? y : fmaxf(x, y)); break;
          case 5: r = isnan(x) ? x : (isnan(y) ? y : fminf(x, y)); break;
        }
        out[i] = r;
      }
      """);

  /** Same op codes as {@link #EW_BINARY}, with the right operand a host scalar. */
  static final GpuKernel EW_SCALAR = GpuKernel.of("""
      extern "C" __global__ void ew_scalar(float* out, const float* a, float s, int n, int op) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        float x = a[i], r = 0.0f;
        switch (op) {
          case 0: r = x + s; break;
          case 1: r = x - s; break;
          case 2: r = x * s; break;
          case 3: r = x / s; break;
          case 4: r = isnan(x) ? x : (isnan(s) ? s : fmaxf(x, s)); break;
          case 5: r = isnan(x) ? x : (isnan(s) ? s : fminf(x, s)); break;
        }
        out[i] = r;
      }
      """);

  /** Op codes must stay in step with the unary cases {@code GpuKernels.unaryExpr} emits. */
  static final GpuKernel EW_UNARY = GpuKernel.of("""
      extern "C" __global__ void ew_unary(float* out, const float* a, int n, int op) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        float x = a[i], r = 0.0f;
        switch (op) {
          case 0: r = -x; break;
          case 1: r = expf(x); break;
          case 2: r = logf(x); break;
          case 3: r = sqrtf(x); break;
          case 4: r = rsqrtf(x); break;
          case 5: r = tanhf(x); break;
          case 6: r = 1.0f / (1.0f + expf(-x)); break;
          case 7: r = isnan(x) ? x : fmaxf(0.0f, x); break;
          case 8: r = fabsf(x); break;
          case 9: r = x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); break;
        }
        out[i] = r;
      }
      """);

  static final GpuKernel RELU_BACKWARD = GpuKernel.of("""
      extern "C" __global__ void relu_backward(float* out, const float* upstream,
          const float* input, int n) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n) return;
        out[index] = input[index] > 0.0f ? upstream[index] : 0.0f;
      }
      """);

  static final List<GpuKernel> KERNELS = List.of(EW_BINARY, EW_SCALAR, EW_UNARY, RELU_BACKWARD);
}
