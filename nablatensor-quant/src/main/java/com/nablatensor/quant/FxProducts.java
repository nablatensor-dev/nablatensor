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
 * FX and quanto options under lognormal dynamics.
 *
 * <ul>
 *   <li><b>FX option</b> (Garman-Kohlhagen): the FX rate drifts at
 *       {@code rateDom - rateForeign}; the run returns delta and both rho's.</li>
 *   <li><b>Quanto option</b>: a payoff on a foreign asset settled in domestic
 *       currency at a fixed FX rate. The foreign asset's domestic-measure drift
 *       carries the quanto adjustment {@code -corr * volAsset * volFx}.</li>
 * </ul>
 */
public final class FxProducts {

  private FxProducts() {
  }

  /** European FX call/put, discounted at the domestic rate. */
  public static BiConsumer<AadRecorder, Nabla.Inputs<FxMarket>> fxOption(
      OptionType type, double maturity, int steps) {
    return (rec, in) -> {
      SDouble x = in.of(FxMarket::spot);
      SDouble strike = in.of(FxMarket::strike);
      SDouble vol = in.of(FxMarket::volFx);
      SDouble rd = in.of(FxMarket::rateDom);
      SDouble rf = in.of(FxMarket::rateForeign);
      double dt = maturity / steps;
      double sqrtDt = Math.sqrt(dt);
      SDouble drift = rd.sub(rf).sub(vol.mul(vol).mul(0.5)).mul(dt);
      for (int t = 0; t < steps; t++) {
        x = x.mul(drift.add(vol.mul(sqrtDt).mul(rec.randn())).exp());
      }
      SDouble intrinsic = type == OptionType.CALL ? x.sub(strike).max(0.0) : strike.sub(x).max(0.0);
      rec.output(intrinsic.mul(rd.neg().mul(maturity).exp()));
    };
  }

  /**
   * Quanto option: payoff {@code fixedFx * max(sign (S_T - K), 0)} in domestic
   * currency, {@code K} and {@code S} in foreign units.
   */
  public static BiConsumer<AadRecorder, Nabla.Inputs<QuantoMarket>> quantoOption(
      OptionType type, double maturity, int steps, double fixedFx) {
    return (rec, in) -> {
      SDouble s = in.of(QuantoMarket::assetSpot);
      SDouble strike = in.of(QuantoMarket::strike);
      SDouble volS = in.of(QuantoMarket::volAsset);
      SDouble volX = in.of(QuantoMarket::volFx);
      SDouble corr = in.of(QuantoMarket::corr);
      SDouble rd = in.of(QuantoMarket::rateDom);
      SDouble rf = in.of(QuantoMarket::rateForeign);
      double dt = maturity / steps;
      double sqrtDt = Math.sqrt(dt);
      // domestic-measure drift of the foreign asset: r_f - corr volS volX - 0.5 volS^2
      SDouble drift = rf.sub(corr.mul(volS).mul(volX)).sub(volS.mul(volS).mul(0.5)).mul(dt);
      for (int t = 0; t < steps; t++) {
        s = s.mul(drift.add(volS.mul(sqrtDt).mul(rec.randn())).exp());
      }
      SDouble intrinsic = type == OptionType.CALL ? s.sub(strike).max(0.0) : strike.sub(s).max(0.0);
      rec.output(intrinsic.mul(fixedFx).mul(rd.neg().mul(maturity).exp()));
    };
  }
}
