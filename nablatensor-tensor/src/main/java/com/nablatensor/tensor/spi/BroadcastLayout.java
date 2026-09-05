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
 * Target dimensions and source strides for a broadcast, both left-padded to
 * {@link #MAX_RANK} with size-1 / stride-0 entries so a kernel can walk them with
 * a fixed-rank loop. An axis that is being broadcast (source extent 1) gets
 * stride 0. Shared by every backend's {@code broadcastTo}, which passes the eight
 * values as kernel scalars.
 *
 * <p>The {@code dims} and {@code strides} arrays are freshly allocated by {@link
 * #of} and belong to the caller.
 */
public record BroadcastLayout(int[] dims, int[] strides) {

  /** Highest tensor rank the {@code broadcast_to} kernels handle. */
  public static final int MAX_RANK = 4;

  /**
   * Builds the padded dims / strides for broadcasting {@code src} to {@code
   * target}. Rejects a target rank above {@link #MAX_RANK}; the caller handles
   * the {@code src.equals(target)} no-op before calling.
   */
  public static BroadcastLayout of(Shape src, Shape target) {
    int rank = target.rank();
    if (rank > MAX_RANK) {
      throw new UnsupportedOperationException(
          "broadcastTo supports up to rank " + MAX_RANK + ", got " + target);
    }
    int pad = rank - src.rank();
    int[] srcDims = new int[rank];
    for (int i = 0; i < rank; i++) {
      srcDims[i] = i < pad ? 1 : src.dim(i - pad);
    }
    int[] srcStrides = new int[rank];
    int stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
      srcStrides[i] = srcDims[i] == 1 ? 0 : stride;
      stride *= srcDims[i];
    }
    int[] targetDims = target.dims();
    int leadPad = MAX_RANK - rank;
    int[] dims = new int[MAX_RANK];
    int[] strides = new int[MAX_RANK];
    for (int i = 0; i < MAX_RANK; i++) {
      dims[i] = i < leadPad ? 1 : targetDims[i - leadPad];
      strides[i] = i < leadPad ? 0 : srcStrides[i - leadPad];
    }
    return new BroadcastLayout(dims, strides);
  }
}
