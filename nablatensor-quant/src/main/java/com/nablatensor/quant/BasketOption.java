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
import java.util.function.BiConsumer;

/**
 * A three-asset arithmetic basket option under correlated GBM.
 *
 * <p>The correlation matrix and the basket weights are fixed model inputs (host
 * Cholesky, done once at record time); the spots, vols and rate are
 * differentiable, so one adjoint sweep returns the per-asset delta and vega
 * vectors of the basket.
 *
 * @see BasketMarket
 */
public final class BasketOption {

  public static final int ASSETS = 3;

  private BasketOption() {
  }

  /**
   * @param type      call or put on the weighted basket level
   * @param weights   basket weights (length 3)
   * @param strike    basket strike
   * @param corr      3×3 correlation matrix
   * @param maturity  years to expiry
   * @param steps     GBM sub-steps
   */
  public static BiConsumer<AadRecorder, Nabla.Inputs<BasketMarket>> option(
      OptionType type, double[] weights, double strike, double[][] corr, double maturity, int steps) {
    if (weights.length != ASSETS || corr.length != ASSETS) {
      throw new IllegalArgumentException("basket is fixed at " + ASSETS + " assets");
    }
    CorrelatedNormals mixer = CorrelatedNormals.of(corr);
    double dt = maturity / steps;
    double sqrtDt = Math.sqrt(dt);

    return (rec, in) -> {
      SDouble rate = in.of(BasketMarket::rate);
      SDouble[] s = {in.of(BasketMarket::s1), in.of(BasketMarket::s2), in.of(BasketMarket::s3)};
      SDouble[] vol = {in.of(BasketMarket::v1), in.of(BasketMarket::v2), in.of(BasketMarket::v3)};
      SDouble[] drift = new SDouble[ASSETS];
      for (int i = 0; i < ASSETS; i++) {
        drift[i] = rate.sub(vol[i].mul(vol[i]).mul(0.5)).mul(dt);
      }
      for (int t = 0; t < steps; t++) {
        SDouble[] z = mixer.draw(rec);
        for (int i = 0; i < ASSETS; i++) {
          s[i] = s[i].mul(drift[i].add(vol[i].mul(sqrtDt).mul(z[i])).exp());
        }
      }
      SDouble level = rec.constant(0.0);
      for (int i = 0; i < ASSETS; i++) {
        level = level.add(s[i].mul(weights[i]));
      }
      SDouble intrinsic = type == OptionType.CALL
          ? level.sub(strike).max(0.0)
          : rec.constant(strike).sub(level).max(0.0);
      rec.output(intrinsic.mul(rate.neg().mul(maturity).exp()));
    };
  }
}
