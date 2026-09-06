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
import com.nablatensor.engine.Nabla;
import com.nablatensor.engine.SDouble;
import java.util.function.BiConsumer;

/**
 * Spread and exchange options on a {@link SpreadMarket}: two correlated GBM legs,
 * a European payoff on their difference, discounted at the flat rate. One
 * adjoint sweep returns both spot deltas and both vegas; the analytic
 * {@link com.nablatensor.quant.analytic.KirkSpreadOption} and
 * {@link com.nablatensor.quant.analytic.Margrabe} are the references.
 *
 * <p>The correlation is a fixed model input (host Cholesky at record time), like
 * {@link BasketOption}.
 */
public final class SpreadProducts {

  private SpreadProducts() {
  }

  /** {@code max(S1_T - S2_T - strike, 0)} discounted; {@code strike = 0} is a Margrabe exchange. */
  public static BiConsumer<AadRecorder, Nabla.Inputs<SpreadMarket>> spreadOption(
      double strike, double rho, double maturity, int steps) {
    return spread(strike, rho, maturity, steps, true);
  }

  /** {@code max(strike - (S1_T - S2_T), 0)} discounted. */
  public static BiConsumer<AadRecorder, Nabla.Inputs<SpreadMarket>> spreadPut(
      double strike, double rho, double maturity, int steps) {
    return spread(strike, rho, maturity, steps, false);
  }

  private static BiConsumer<AadRecorder, Nabla.Inputs<SpreadMarket>> spread(
      double strike, double rho, double maturity, int steps, boolean call) {
    CorrelatedNormals mix = CorrelatedNormals.pair(rho);
    double dt = maturity / steps;
    double sqrtDt = Math.sqrt(dt);
    return (rec, in) -> {
      SDouble s1 = in.of(SpreadMarket::s1);
      SDouble s2 = in.of(SpreadMarket::s2);
      SDouble v1 = in.of(SpreadMarket::vol1);
      SDouble v2 = in.of(SpreadMarket::vol2);
      SDouble r = in.of(SpreadMarket::rate);
      SDouble q1 = in.of(SpreadMarket::yield1);
      SDouble q2 = in.of(SpreadMarket::yield2);
      SDouble drift1 = r.sub(q1).sub(v1.mul(v1).mul(0.5)).mul(dt);
      SDouble drift2 = r.sub(q2).sub(v2.mul(v2).mul(0.5)).mul(dt);
      for (int t = 0; t < steps; t++) {
        SDouble[] z = mix.draw(rec);
        s1 = s1.mul(drift1.add(v1.mul(sqrtDt).mul(z[0])).exp());
        s2 = s2.mul(drift2.add(v2.mul(sqrtDt).mul(z[1])).exp());
      }
      SDouble spread = s1.sub(s2);
      SDouble intrinsic = call
          ? spread.sub(strike).max(0.0)
          : rec.constant(strike).sub(spread).max(0.0);
      rec.output(intrinsic.mul(r.neg().mul(maturity).exp()));
    };
  }
}
