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
package com.nablatensor.quant.adjust;

import com.nablatensor.quant.OptionType;
import com.nablatensor.quant.QuantoMarket;
import com.nablatensor.quant.analytic.AnalyticGreeks;
import com.nablatensor.quant.analytic.Black76;

/**
 * The quanto adjustment: a payoff on a foreign asset that settles in domestic
 * currency at a fixed FX rate. Changing to the domestic risk-neutral measure
 * subtracts {@code corr * volAsset * volFx} from the foreign asset's growth
 * rate, so its quanto forward is
 * {@code S_0 exp((rateForeign - corr volAsset volFx) T)} and a quanto option is
 * a {@link Black76} on that forward, discounted at the domestic rate and scaled
 * by the fixed FX.
 */
public final class QuantoAdjustment {

  private QuantoAdjustment() {
  }

  /** The drift adjustment {@code -corr * volAsset * volFx} applied in the domestic measure. */
  public static double driftAdjustment(double corr, double volAsset, double volFx) {
    return -corr * volAsset * volFx;
  }

  /** Quanto forward of the foreign asset: {@code S_0 exp((rateForeign - corr volAsset volFx) T)}. */
  public static double quantoForward(QuantoMarket m, double maturity) {
    return m.assetSpot() * Math.exp((m.rateForeign() + driftAdjustment(m.corr(), m.volAsset(), m.volFx()))
        * maturity);
  }

  /**
   * Quanto option value in domestic currency:
   * {@code fixedFx * e^{-rateDom T} * Black(F_quanto, K, T, volAsset)}.
   */
  public static AnalyticGreeks quantoOption(OptionType type, QuantoMarket m, double maturity, double fixedFx) {
    double fq = quantoForward(m, maturity);
    AnalyticGreeks black = Black76.of(type, fq, m.strike(), maturity, m.rateDom(), m.volAsset());
    return new AnalyticGreeks(
        fixedFx * black.price(),
        fixedFx * black.delta(),
        fixedFx * black.gamma(),
        fixedFx * black.vega(),
        fixedFx * black.theta(),
        fixedFx * black.rho(),
        fixedFx * black.strikeSensitivity());
  }
}
