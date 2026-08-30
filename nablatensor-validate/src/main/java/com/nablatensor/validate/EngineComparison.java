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
package com.nablatensor.validate;

import com.nablatensor.quant.EquityMarket;
import com.nablatensor.engine.Nabla;

/**
 * One backend's replay of a tape, diffed against the scalar CPU oracle at an
 * equal seed and scenario count.
 *
 * <p>{@code priceAbsDiff} and {@code gradMaxAbsDiff} are absolute; the relative
 * figures are divided by {@code 1 + |oracle|} so an oracle value near zero does
 * not blow the ratio up.
 */
public record EngineComparison(String engine, String describe,
                               double priceAbsDiff, double priceRelDiff,
                               double gradMaxAbsDiff, double gradMaxRelDiff,
                               boolean withinTolerance) {

  static EngineComparison of(String engine, String describe,
                             Nabla.TypedValuation<EquityMarket> oracle,
                             Nabla.TypedValuation<EquityMarket> candidate,
                             double tolerance) {
    double priceAbs = Math.abs(oracle.price() - candidate.price());
    double priceRel = priceAbs / (1.0 + Math.abs(oracle.price()));

    double gradAbs = 0.0;
    double gradRel = 0.0;
    {
      EquityMarket a = oracle.greeks();
      EquityMarket b = candidate.greeks();
      double[] o = {a.spot(), a.strike(), a.vol(), a.rate(), a.maturity()};
      double[] c = {b.spot(), b.strike(), b.vol(), b.rate(), b.maturity()};
      for (int i = 0; i < o.length; i++) {
        double abs = Math.abs(o[i] - c[i]);
        gradAbs = Math.max(gradAbs, abs);
        gradRel = Math.max(gradRel, abs / (1.0 + Math.abs(o[i])));
      }
    }
    boolean ok = priceRel <= tolerance && gradRel <= tolerance;
    return new EngineComparison(engine, describe, priceAbs, priceRel, gradAbs, gradRel, ok);
  }
}
