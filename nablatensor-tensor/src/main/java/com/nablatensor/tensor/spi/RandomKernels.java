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
 * Counter-based random kernels: a draw is a pure function of {@code (seed,
 * counter + i)}, so a run reproduces without any device-side RNG state. Both
 * call the mixing helpers in {@link DevicePrelude}.
 */
final class RandomKernels {

  private RandomKernels() {
  }

  static final GpuKernel RANDOM_UNIFORM = GpuKernel.of("""
      extern "C" __global__ void random_uniform(
          float* out, unsigned long long seed, unsigned long long counter, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = random_uniform_value(seed, counter + (unsigned long long) i);
      }
      """);

  /** Box-Muller on two draws per thread; {@code u1} is nudged off zero before {@code logf}. */
  static final GpuKernel RANDOM_NORMAL = GpuKernel.of("""
      extern "C" __global__ void random_normal(
          float* out, unsigned long long seed, unsigned long long counter, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        unsigned long long first = counter + 2ULL * (unsigned long long) i;
        unsigned long long bits = random_mix64(seed + 0x9E3779B97F4A7C15ULL * first);
        float u1 = (float) ((bits >> 40) + 1ULL) * 5.9604644775390625e-8f;
        float u2 = random_uniform_value(seed, first + 1ULL);
        out[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853071795864769f * u2);
      }
      """);

  static final List<GpuKernel> KERNELS = List.of(RANDOM_UNIFORM, RANDOM_NORMAL);
}
