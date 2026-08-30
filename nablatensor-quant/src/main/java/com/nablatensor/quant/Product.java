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
package com.nablatensor.quant;

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.SDouble;
import com.nablatensor.engine.Nabla;

/**
 * A payoff recorded against a market record {@code M} (Seam 1). {@code M} is
 * {@link EquityMarket} for the built-in equity catalogue; any {@code double}-only
 * record works, so a rates or FX payoff records against its own market type
 * through the same interface.
 *
 * <p>An implementation writes the valuation in plain Java over {@link SDouble}
 * scalars and calls {@code rec.output(...)} exactly once. It never touches a
 * device or a Greek: the engine records the tape and one adjoint sweep produces
 * every sensitivity.
 *
 * <p>The built-ins live in {@link Products}. To price something the catalogue
 * does not cover, pass a lambda of this shape straight to
 * {@link MonteCarlo#of(Product)} — changing a payoff is a three-line diff, not a
 * fork.
 *
 * @param <M> the market record this payoff reads its differentiable inputs from
 */
@FunctionalInterface
public interface Product<M extends Record> {

  /**
   * Records the discounted payoff.
   *
   * @param rec  the recorder in progress
   * @param in   the market inputs, read by accessor: {@code in.of(EquityMarket::spot)}
   * @param grid the simulation schedule; {@code grid.steps()} time steps, step
   *             {@code i} spanning {@code maturity * grid.fraction(i)}
   */
  void record(AadRecorder rec, Nabla.Inputs<M> in, TimeGrid grid);

  /** Short label used in reports and example output. */
  default String label() {
    return getClass().getSimpleName();
  }
}
