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

/**
 * {@code __device__} helpers emitted ahead of every kernel in the shared
 * translation unit, so more than one kernel family can call them.
 */
final class DevicePrelude {

  private DevicePrelude() {
  }

  /** SplitMix64 bit mixing, used by the counter-based random kernels. */
  static final String SOURCE = """
      __device__ __forceinline__ unsigned long long random_mix64(unsigned long long value) {
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31);
      }
      __device__ __forceinline__ float random_uniform_value(
          unsigned long long seed, unsigned long long counter) {
        unsigned long long bits =
            random_mix64(seed + 0x9E3779B97F4A7C15ULL * counter);
        return (float) (bits >> 40) * 5.9604644775390625e-8f;
      }
      """;
}
