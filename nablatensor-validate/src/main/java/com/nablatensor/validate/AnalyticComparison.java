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
import com.nablatensor.quant.analytic.AnalyticGreeks;
import com.nablatensor.engine.Nabla;

/**
 * The scalar oracle's adjoint price and gradient diffed against an independent
 * closed form (feature F2). Where the bump cross-check confirms the adjoint
 * against a <em>numerical</em> derivative of the same Monte-Carlo estimator,
 * this confirms it against a formula that shares no code with the simulation —
 * the stronger evidence, when a closed form exists for the contract.
 *
 * <p>The differences are Monte-Carlo error plus discretisation error, so the
 * tolerance is looser than the backend-reproduction check; {@code withinBand}
 * uses {@code 3 * standardError + 5e-3 * |reference|} per quantity.
 */
public record AnalyticComparison(AnalyticGreeks reference,
                                 double priceAbsDiff, double deltaAbsDiff, double vegaAbsDiff,
                                 double rhoAbsDiff, double strikeAbsDiff,
                                 boolean withinBand) {

  static AnalyticComparison of(Nabla.TypedValuation<EquityMarket> oracle, AnalyticGreeks reference) {
    EquityMarket g = oracle.greeks();
    double se = Math.max(oracle.standardError(), 0.0);

    double dPrice = Math.abs(oracle.price() - reference.price());
    double dDelta = Math.abs(g.spot() - reference.delta());
    double dVega = Math.abs(g.vol() - reference.vega());
    double dRho = Math.abs(g.rate() - reference.rho());
    double dStrike = Math.abs(g.strike() - reference.strikeSensitivity());

    boolean ok = within(dPrice, se, reference.price())
        && within(dDelta, se, reference.delta())
        && within(dVega, se, reference.vega())
        && within(dRho, se, reference.rho())
        && within(dStrike, se, reference.strikeSensitivity());

    return new AnalyticComparison(reference, dPrice, dDelta, dVega, dRho, dStrike, ok);
  }

  private static boolean within(double diff, double standardError, double reference) {
    return diff <= 3.0 * standardError + 5e-3 * Math.max(1.0, Math.abs(reference));
  }
}
