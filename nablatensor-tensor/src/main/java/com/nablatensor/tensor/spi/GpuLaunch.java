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
 * Launch-geometry arithmetic shared by the CUDA, ROCm, and Vulkan compute
 * backends: how many blocks / workgroups cover a flat element count. Kept in one
 * place so the formula (and the block size the flat one-dimensional kernels are
 * written for) has a single definition.
 */
public final class GpuLaunch {

  private GpuLaunch() {
  }

  /** Threads per block / workgroup the flat one-dimensional kernels are written for. */
  public static final int DEFAULT_BLOCK = 256;

  /** Blocks needed to cover {@code elements} at {@link #DEFAULT_BLOCK} threads each. */
  public static int grid1d(int elements) {
    return grid1d(elements, DEFAULT_BLOCK);
  }

  /** Blocks needed to cover {@code elements} at {@code block} threads each; {@code 0 -> 0}. */
  public static int grid1d(int elements, int block) {
    return elements == 0 ? 0 : 1 + (elements - 1) / block;
  }

  /** {@code ceil(size / divisor)} for a positive {@code divisor}. */
  public static int ceilDiv(int size, int divisor) {
    return (size + divisor - 1) / divisor;
  }
}
