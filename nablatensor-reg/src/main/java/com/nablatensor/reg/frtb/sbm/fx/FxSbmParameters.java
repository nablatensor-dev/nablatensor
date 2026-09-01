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
package com.nablatensor.reg.frtb.sbm.fx;

import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskClassProfile;
import com.nablatensor.risk.RiskFactor;
import java.util.Set;

/**
 * FRTB SA-SBM <b>foreign-exchange</b> risk-class parameters, from the Basel
 * Framework MAR21:
 *
 * <ul>
 *   <li>one delta risk factor per currency pair (reporting currency vs each
 *       other currency); delta risk weight {@code 15%};</li>
 *   <li>for the specified liquid currency pairs the risk weight is
 *       {@code 15% / sqrt(2)}; this profile applies the relief when <b>both</b>
 *       legs of the pair are Basel-specified liquid currencies (a simplification
 *       of the "against a common third currency" wording — verify);</li>
 *   <li>one factor per bucket, so within-bucket correlation is {@code 1};
 *       across-bucket correlation {@code gamma = 60%};</li>
 *   <li>curvature shock: relative shock of the exchange rate by the FX risk
 *       weight;</li>
 *   <li>vega risk weight {@code min(0.55*sqrt(LH/10), 1)}, {@code LH = 40}.</li>
 * </ul>
 *
 * <p>This is a data table; check it against your regulator's current rulebook.
 */
public final class FxSbmParameters implements RiskClassProfile {

  private static final double BASE_RW = 0.15;
  private static final double GAMMA = 0.60;
  private static final double VEGA_LH = 40.0;
  private static final double VEGA_ALPHA = 0.01;

  private static final Set<String> LIQUID_CURRENCIES =
      Set.of("USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "MXN", "CNY", "NZD",
             "RUB", "HKD", "SGD", "TRY", "KRW", "SEK", "ZAR", "INR", "NOK", "BRL");

  private FxSbmParameters() {
  }

  public static final FxSbmParameters INSTANCE = new FxSbmParameters();

  @Override
  public RiskClass riskClass() {
    return RiskClass.FX;
  }

  @Override
  public double deltaRiskWeight(RiskFactor k) {
    return reliefApplies(k.name()) ? BASE_RW / Math.sqrt(2.0) : BASE_RW;
  }

  @Override
  public double vegaRiskWeight(RiskFactor k) {
    return Math.min(0.55 * Math.sqrt(VEGA_LH / 10.0), 1.0);
  }

  @Override
  public double curvatureShock(RiskFactor k, double riskFactorLevel) {
    return deltaRiskWeight(k) * riskFactorLevel;
  }

  @Override
  public double deltaRho(RiskFactor k, RiskFactor l) {
    return k.equals(l) ? 1.0 : 0.0;   // one factor per bucket
  }

  @Override
  public double vegaRho(RiskFactor k, RiskFactor l) {
    if (k.equals(l)) {
      return 1.0;
    }
    if (!k.bucket().equals(l.bucket())) {
      return 0.0;
    }
    double x = Math.max(k.tenor(), 1e-9);
    double y = Math.max(l.tenor(), 1e-9);
    return Math.exp(-VEGA_ALPHA * Math.abs(x - y) / Math.min(x, y));
  }

  @Override
  public double gamma(String bucketB, String bucketC) {
    return bucketB.equals(bucketC) ? 1.0 : GAMMA;
  }

  private static boolean reliefApplies(String currencyPair) {
    if (currencyPair.length() != 6) {
      return false;
    }
    String a = currencyPair.substring(0, 3);
    String b = currencyPair.substring(3, 6);
    return LIQUID_CURRENCIES.contains(a) && LIQUID_CURRENCIES.contains(b);
  }
}
