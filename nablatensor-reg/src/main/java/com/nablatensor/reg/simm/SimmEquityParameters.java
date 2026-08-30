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
package com.nablatensor.reg.simm;

import com.nablatensor.risk.RiskFactor;

/**
 * ISDA SIMM <b>equity</b> parameters. SIMM is recalibrated annually and its risk
 * weights, correlations and concentration thresholds are version-specific and
 * licence-restricted; the values here are <b>illustrative placeholders with the
 * correct structure</b>. Replace them with the current ISDA-published set (the
 * SIMM "unit test" / calibration file) for any real use — {@link IsdaSimm} is
 * fully parameter-driven so only this class changes.
 *
 * <p>Structure that is <em>not</em> a placeholder: {@code WS_k = RW_b s_k CR_k}
 * with {@code CR_k = max(1, sqrt(|s_k| / T_b))}, the within-bucket factor
 * {@code f_kl = min(CR_k,CR_l)/max(CR_k,CR_l)}, and the across-bucket factor
 * {@code g_bc = min(CR_b,CR_c)/max(CR_b,CR_c)}.
 */
public final class SimmEquityParameters {

  // ---- ILLUSTRATIVE VALUES (not the ISDA calibration) --------------------
  private static final double[] DELTA_RW = {
      Double.NaN, 25, 32, 29, 27, 18, 21, 25, 22, 27, 29, 16, 16
  };
  private static final double[] DELTA_THRESHOLD_MM = {   // USD millions
      Double.NaN, 8, 8, 8, 8, 26, 26, 26, 26, 8, 26, 8, 8
  };
  private static final double[] WITHIN_RHO = {
      Double.NaN, 0.14, 0.20, 0.19, 0.21, 0.24, 0.35, 0.34, 0.34, 0.20, 0.24, 0.00, 0.50
  };
  private static final double VEGA_RW = 0.28;
  private static final double VEGA_THRESHOLD_MM = 210;
  private static final double HVR = 0.60;   // historical volatility ratio applied to vega

  private SimmEquityParameters() {
  }

  static int bucket(RiskFactor f) {
    return Integer.parseInt(f.bucket());
  }

  public static double deltaRiskWeight(RiskFactor f) {
    return DELTA_RW[bucket(f)] / 100.0;
  }

  public static double deltaConcentrationThreshold(RiskFactor f) {
    return DELTA_THRESHOLD_MM[bucket(f)] * 1e6;
  }

  public static double vegaRiskWeight() {
    return VEGA_RW;
  }

  public static double vegaConcentrationThreshold() {
    return VEGA_THRESHOLD_MM * 1e6;
  }

  public static double historicalVolatilityRatio() {
    return HVR;
  }

  public static double withinBucketRho(RiskFactor k, RiskFactor l) {
    if (k.equals(l)) {
      return 1.0;
    }
    int b = bucket(k);
    return b == bucket(l) ? WITHIN_RHO[b] : 0.0;
  }

  public static double acrossBucketGamma(String b, String c) {
    if (b.equals(c)) {
      return 1.0;
    }
    if (b.equals("11") || c.equals("11")) {
      return 0.0;
    }
    return 0.15;   // illustrative flat cross-bucket correlation
  }
}
