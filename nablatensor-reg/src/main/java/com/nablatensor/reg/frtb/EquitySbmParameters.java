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
package com.nablatensor.reg.frtb;

import com.nablatensor.risk.RiskFactor;

/**
 * FRTB SA-SBM <b>equity</b> risk-class parameters, from the Basel Framework
 * MAR21 (BCBS d457). Values as published for the equity spot price factor:
 *
 * <ul>
 *   <li>delta risk weights by bucket 1..13:
 *       {@code 55 60 45 55 | 30 35 40 50 | 70 50 | 70 | 15 25} percent</li>
 *   <li>within-bucket delta correlation (different names): buckets 1-4 = 0.15,
 *       5-8 = 0.25, 9 = 0.075, 10 = 0.125, 11 = 0 (factors stand alone),
 *       12-13 = 0.80; spot vs repo of the same issuer = 0.999</li>
 *   <li>across-bucket correlation gamma: both buckets in 1-10 = 0.15;
 *       either bucket is 11 = 0; buckets 12 and 13 = 0.75; otherwise = 0.45</li>
 *   <li>curvature risk weight per bucket = that bucket's delta risk weight</li>
 *   <li>vega risk weight = {@code min(0.55 * sqrt(LH/10), 1)}, LH = 20 (large
 *       cap, buckets 1-8 &amp; 12) or 120 (small cap / other, buckets 9-11 &amp; 13)</li>
 * </ul>
 *
 * <p>This is a data table; check it against your regulator's current rulebook.
 */
public final class EquitySbmParameters {

  private static final double[] DELTA_RW = {
      Double.NaN,                     // 1-indexed
      0.55, 0.60, 0.45, 0.55,
      0.30, 0.35, 0.40, 0.50,
      0.70, 0.50,
      0.70,
      0.15, 0.25
  };

  private EquitySbmParameters() {
  }

  public static int bucket(RiskFactor f) {
    return Integer.parseInt(f.bucket());
  }

  public static double deltaRiskWeight(RiskFactor f) {
    return DELTA_RW[bucket(f)];
  }

  public static double curvatureRiskWeight(int bucket) {
    return DELTA_RW[bucket];
  }

  public static double vegaRiskWeight(RiskFactor f) {
    int b = bucket(f);
    double lh = (b <= 8 || b == 12) ? 20.0 : 120.0;
    return Math.min(0.55 * Math.sqrt(lh / 10.0), 1.0);
  }

  /** Within-bucket delta correlation between two distinct names in the same bucket. */
  public static double deltaRho(RiskFactor k, RiskFactor l) {
    if (k.equals(l)) {
      return 1.0;
    }
    int b = bucket(k);
    if (b != bucket(l)) {
      return 0.0;
    }
    return switch (b) {
      case 1, 2, 3, 4 -> 0.15;
      case 5, 6, 7, 8 -> 0.25;
      case 9 -> 0.075;
      case 10 -> 0.125;
      case 11 -> 0.0;
      case 12, 13 -> 0.80;
      default -> throw new IllegalArgumentException("equity bucket " + b);
    };
  }

  /** Vega within-bucket correlation: delta correlation times the option-maturity factor. */
  public static double vegaRho(RiskFactor k, RiskFactor l) {
    if (k.equals(l)) {
      return 1.0;
    }
    double base = deltaRho(new RiskFactor(k.riskClass(), com.nablatensor.risk.RiskMeasure.DELTA, k.bucket(), k.name()),
        new RiskFactor(l.riskClass(), com.nablatensor.risk.RiskMeasure.DELTA, l.bucket(), l.name()));
    double tk = Math.max(k.tenor(), 1e-9);
    double tl = Math.max(l.tenor(), 1e-9);
    double maturity = Math.exp(-0.01 * Math.abs(tk - tl) / Math.min(tk, tl));
    return base * maturity;
  }

  public static double gamma(String bucketB, String bucketC) {
    int b = Integer.parseInt(bucketB);
    int c = Integer.parseInt(bucketC);
    if (b == c) {
      return 1.0;
    }
    if (b == 11 || c == 11) {
      return 0.0;
    }
    if ((b == 12 && c == 13) || (b == 13 && c == 12)) {
      return 0.75;
    }
    boolean bIndex = b >= 12;
    boolean cIndex = c >= 12;
    if (bIndex != cIndex) {
      return 0.45;
    }
    return 0.15;   // both in 1-10
  }
}
