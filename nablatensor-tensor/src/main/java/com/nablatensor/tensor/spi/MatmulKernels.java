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
 * Layout and GEMM kernels. The two tiled kernels use a 16x16 thread block that
 * matches their {@code __shared__} tiles, and take a {@code tile_offset} so a
 * grid wider than {@code INT_MAX} blocks can be launched in chunks.
 */
final class MatmulKernels {

  private MatmulKernels() {
  }

  /** Tile edge of the tiled GEMMs; the block shape and the {@code __shared__} tiles must agree. */
  static final int TILE = 16;

  static final GpuKernel TRANSPOSE2D = GpuKernel.of("""
      extern "C" __global__ void transpose2d(float* out, const float* in, int rows, int cols) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= rows * cols) return;
        int r = i / cols, c = i % cols;
        out[c * rows + r] = in[i];
      }
      """);

  static final GpuKernel MATMUL_TILED = GpuKernel.of("""
      extern "C" __global__ void matmul_tiled(
          float* out, const float* a, const float* b, int M, int K, int N,
          unsigned long long tile_offset) {
        __shared__ float tile_a[16][16];
        __shared__ float tile_b[16][16];
        unsigned long long global_tile = tile_offset + (unsigned long long) blockIdx.x;
        unsigned long long column_tiles = ((unsigned long long) N + 15ULL) / 16ULL;
        unsigned long long tile_row = global_tile / column_tiles;
        unsigned long long tile_col = global_tile % column_tiles;
        unsigned long long row = tile_row * 16ULL + threadIdx.y;
        unsigned long long col = tile_col * 16ULL + threadIdx.x;
        float acc = 0.0f;
        unsigned long long k_tiles = ((unsigned long long) K + 15ULL) / 16ULL;
        for (unsigned long long tile = 0; tile < k_tiles; tile++) {
          unsigned long long a_col = tile * 16ULL + threadIdx.x;
          unsigned long long b_row = tile * 16ULL + threadIdx.y;
          tile_a[threadIdx.y][threadIdx.x] =
              row < (unsigned long long) M && a_col < (unsigned long long) K
                  ? a[row * (unsigned long long) K + a_col] : 0.0f;
          tile_b[threadIdx.y][threadIdx.x] =
              b_row < (unsigned long long) K && col < (unsigned long long) N
                  ? b[b_row * (unsigned long long) N + col] : 0.0f;
          __syncthreads();
          #pragma unroll
          for (int p = 0; p < 16; p++) {
            acc += tile_a[threadIdx.y][p] * tile_b[p][threadIdx.x];
          }
          __syncthreads();
        }
        if (row < (unsigned long long) M && col < (unsigned long long) N) {
          out[row * (unsigned long long) N + col] = acc;
        }
      }
      """, TILE, TILE);

  static final GpuKernel BATCHED_MATMUL_TILED = GpuKernel.of("""
      extern "C" __global__ void batched_matmul_tiled(
          float* out, const float* a, const float* b, int batch, int M, int K, int N,
          unsigned long long tile_offset) {
        __shared__ float tile_a[16][16];
        __shared__ float tile_b[16][16];
        unsigned long long global_tile = tile_offset + (unsigned long long) blockIdx.x;
        unsigned long long row_tiles = ((unsigned long long) M + 15ULL) / 16ULL;
        unsigned long long column_tiles = ((unsigned long long) N + 15ULL) / 16ULL;
        unsigned long long tiles_per_batch = row_tiles * column_tiles;
        unsigned long long batch_index = global_tile / tiles_per_batch;
        unsigned long long batch_tile = global_tile % tiles_per_batch;
        unsigned long long row = (batch_tile / column_tiles) * 16ULL + threadIdx.y;
        unsigned long long col = (batch_tile % column_tiles) * 16ULL + threadIdx.x;
        unsigned long long a_offset = batch_index * (unsigned long long) M * K;
        unsigned long long b_offset = batch_index * (unsigned long long) K * N;
        unsigned long long out_offset = batch_index * (unsigned long long) M * N;
        float acc = 0.0f;
        unsigned long long k_tiles = ((unsigned long long) K + 15ULL) / 16ULL;
        for (unsigned long long tile = 0; tile < k_tiles; tile++) {
          unsigned long long a_col = tile * 16ULL + threadIdx.x;
          unsigned long long b_row = tile * 16ULL + threadIdx.y;
          tile_a[threadIdx.y][threadIdx.x] =
              row < (unsigned long long) M && a_col < (unsigned long long) K
                  ? a[a_offset + row * (unsigned long long) K + a_col] : 0.0f;
          tile_b[threadIdx.y][threadIdx.x] =
              b_row < (unsigned long long) K && col < (unsigned long long) N
                  ? b[b_offset + b_row * (unsigned long long) N + col] : 0.0f;
          __syncthreads();
          #pragma unroll
          for (int p = 0; p < 16; p++) {
            acc += tile_a[threadIdx.y][p] * tile_b[p][threadIdx.x];
          }
          __syncthreads();
        }
        if (batch_index < (unsigned long long) batch
            && row < (unsigned long long) M && col < (unsigned long long) N) {
          out[out_offset + row * (unsigned long long) N + col] = acc;
        }
      }
      """, TILE, TILE);

  static final List<GpuKernel> KERNELS = List.of(TRANSPOSE2D, MATMUL_TILED, BATCHED_MATMUL_TILED);
}
