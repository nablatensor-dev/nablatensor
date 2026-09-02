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

/**
 * Reductions and the shape-changing kernels that pair with them. The whole-tensor
 * reductions run as a single block and tree-reduce in shared memory; the axis
 * reductions run one thread per output element and walk the axis serially.
 */
final class ReductionKernels {

  private ReductionKernels() {
  }

  /** The {@code __shared__} scratch is sized for this block, so it is not tunable per launch. */
  static final GpuKernel REDUCE_SUM = GpuKernel.of("""
      extern "C" __global__ void reduce_sum(float* out, const float* a, int n) {
        __shared__ float sdata[256];
        int tid = threadIdx.x;
        float acc = 0.0f;
        for (int i = tid; i < n; i += blockDim.x) acc += a[i];
        sdata[tid] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
          if (tid < s) sdata[tid] += sdata[tid + s];
          __syncthreads();
        }
        if (tid == 0) out[0] = sdata[0];
      }
      """);

  static final GpuKernel REDUCE_MAX = GpuKernel.of("""
      extern "C" __global__ void reduce_max(float* out, const float* a, int n) {
        __shared__ float sdata[256];
        int tid = threadIdx.x;
        float best = -3.402823466e+38f;
        for (int i = tid; i < n; i += blockDim.x) { float v = a[i]; if (v > best) best = v; }
        sdata[tid] = best;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
          if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
          __syncthreads();
        }
        if (tid == 0) out[0] = sdata[0];
      }
      """);

  static final GpuKernel SUM_AXIS0 = GpuKernel.of("""
      extern "C" __global__ void sum_axis0(float* out, const float* a, int rows, int cols) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= cols) return;
        float acc = 0.0f;
        for (int r = 0; r < rows; r++) acc += a[r * cols + c];
        out[c] = acc;
      }
      """);

  static final GpuKernel REDUCE_AXIS_SUM = GpuKernel.of("""
      extern "C" __global__ void reduce_axis_sum(
          float* out, const float* a, int outer, int axis_size, int inner) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= outer * inner) return;
        int outer_index = index / inner;
        int inner_index = index % inner;
        int base = outer_index * axis_size * inner + inner_index;
        float total = 0.0f;
        for (int i = 0; i < axis_size; i++) total += a[base + i * inner];
        out[index] = total;
      }
      """);

  /** NaN wins over any value, matching the elementwise {@code fmaxf} guards. */
  static final GpuKernel REDUCE_AXIS_MAX = GpuKernel.of("""
      extern "C" __global__ void reduce_axis_max(
          float* out, const float* a, int outer, int axis_size, int inner) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= outer * inner) return;
        int outer_index = index / inner;
        int inner_index = index % inner;
        int base = outer_index * axis_size * inner + inner_index;
        float best = -1.0f / 0.0f;
        for (int i = 0; i < axis_size; i++) {
          float value = a[base + i * inner];
          if (isnan(value) || value > best) best = value;
        }
        out[index] = best;
      }
      """);

  static final GpuKernel REDUCE_AXIS_ARGMAX = GpuKernel.of("""
      extern "C" __global__ void reduce_axis_argmax(
          float* out, const float* a, int outer, int axis_size, int inner) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= outer * inner) return;
        int outer_index = index / inner;
        int inner_index = index % inner;
        int base = outer_index * axis_size * inner + inner_index;
        float best = -3.402823466e+38f;
        int best_index = 0;
        for (int i = 0; i < axis_size; i++) {
          float value = a[base + i * inner];
          if (value > best) {
            best = value;
            best_index = i;
          }
        }
        out[index] = (float) best_index;
      }
      """);

  /** Routes the upstream gradient to the winning slot only, re-deriving the winner. */
  static final GpuKernel REDUCE_AXIS_MAX_BACKWARD = GpuKernel.of("""
      extern "C" __global__ void reduce_axis_max_backward(
          float* out, const float* upstream, const float* input,
          int outer, int axis_size, int inner) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= outer * inner) return;
        int outer_index = index / inner;
        int inner_index = index % inner;
        int base = outer_index * axis_size * inner + inner_index;
        float best = -1.0f / 0.0f;
        int best_index = 0;
        for (int i = 0; i < axis_size; i++) {
          float value = input[base + i * inner];
          if (isnan(value) || value > best) {
            best = value;
            best_index = i;
            if (isnan(value)) break;
          }
        }
        for (int i = 0; i < axis_size; i++) {
          out[base + i * inner] = i == best_index ? upstream[index] : 0.0f;
        }
      }
      """);

  /** Rank is fixed at 4; callers left-pad shorter shapes with size-1, stride-0 dims. */
  static final GpuKernel BROADCAST_TO = GpuKernel.of("""
      extern "C" __global__ void broadcast_to(float* out, const float* in,
          int d0, int d1, int d2, int d3, int s0, int s1, int s2, int s3, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int dims[4] = {d0, d1, d2, d3};
        int strides[4] = {s0, s1, s2, s3};
        long tstride[4];
        long acc = 1;
        for (int d = 3; d >= 0; d--) { tstride[d] = acc; acc *= dims[d]; }
        long rem = i;
        long srcOffset = 0;
        for (int d = 0; d < 4; d++) {
          long coord = rem / tstride[d];
          rem = rem % tstride[d];
          srcOffset += coord * strides[d];
        }
        out[i] = in[srcOffset];
      }
      """);

  static final List<GpuKernel> KERNELS = List.of(
      REDUCE_SUM, REDUCE_MAX, SUM_AXIS0, REDUCE_AXIS_SUM, REDUCE_AXIS_MAX,
      REDUCE_AXIS_ARGMAX, REDUCE_AXIS_MAX_BACKWARD, BROADCAST_TO);
}
