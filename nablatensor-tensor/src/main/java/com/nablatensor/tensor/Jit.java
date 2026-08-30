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
package com.nablatensor.tensor;

import com.nablatensor.tensor.spi.ComputeBackend;
import com.nablatensor.tensor.spi.DeviceBuffer;

import java.util.function.UnaryOperator;

/**
 * Traces a fluent {@code Tensor} function and fuses each maximal elementwise
 * chain (add/sub/mul/div/unary/scalar ops) between matmul/transpose
 * boundaries into a single backend kernel launch, instead of one launch per
 * op. Tracing itself is cheap (plain object graph building); backends cache
 * compiled fused kernels by their generated source, so repeated calls with
 * the same op chain and shape reuse the compiled kernel.
 */
public final class Jit {

  private Jit() {
  }

  public static Tensor apply(UnaryOperator<Tensor> body, Tensor input) {
    ComputeBackend real = input.backend();
    FusingBackend fusing = new FusingBackend(real);
    Tensor traced = new Tensor(fusing, fusing.leaf(input.buffer()));
    Tensor output = body.apply(traced);
    DeviceBuffer materialized = fusing.materialize(output.buffer());
    fusing.releaseIntermediates(materialized);
    return new Tensor(real, materialized);
  }
}
