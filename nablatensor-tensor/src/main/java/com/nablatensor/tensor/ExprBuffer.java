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

import com.nablatensor.tensor.expr.Expr;
import com.nablatensor.tensor.spi.DeviceBuffer;

import java.util.List;

/**
 * A lazily-fused elementwise chain traced by {@link Jit}: {@code expr}
 * describes the pending computation and {@code leaves} are the real backend
 * buffers it reads from. Real compute is deferred until a fusion boundary
 * (matmul/transpose/download) forces materialization.
 */
record ExprBuffer(Expr expr, List<DeviceBuffer> leaves, Shape shape, DType dtype, Device device)
    implements DeviceBuffer {
}
