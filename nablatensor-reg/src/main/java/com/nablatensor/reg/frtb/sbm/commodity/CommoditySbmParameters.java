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
package com.nablatensor.reg.frtb.sbm.commodity;

import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskClassProfile;
import com.nablatensor.risk.RiskFactor;

/**
 * FRTB SA-SBM <b>commodity</b> risk-class parameters, from the Basel Framework
 * MAR21:
 *
 * <ul>
 *   <li>11 buckets; delta risk weights {@code 30 35 60 80 40 45 20 35 25 35 50}
 *       percent (MAR21.82);</li>
 *   <li>within-bucket correlation {@code rho_commodity * rho_tenor *
 *       rho_location}: {@code rho_commodity} per bucket (1 if the same
 *       commodity), {@code rho_tenor = 99%}, {@code rho_location = 99.9%};</li>
 *   <li>across-bucket correlation {@code gamma = 20%} (MAR21.84 — a
 *       simplification of the published pairwise values);</li>
 *   <li>curvature shock: relative parallel shift of the forward curve by the
 *       bucket risk weight;</li>
 *   <li>vega risk weight {@code min(0.55*sqrt(LH/10), 1)}, {@code LH = 120}.</li>
 * </ul>
 *
 * <p>This is a data table; check it against your regulator's current rulebook.
 */
public final class CommoditySbmParameters implements RiskClassProfile {

  private static final double[] DELTA_RW = {
      Double.NaN, 0.30, 0.35, 0.60, 0.80, 0.40, 0.45, 0.20, 0.35, 0.25, 0.35, 0.50
  };
  private static final double[] RHO_COMMODITY = {
      Double.NaN, 0.55, 0.95, 0.40, 0.80, 0.60, 0.65, 0.55, 0.45, 0.15, 0.40, 0.15
  };
  private static final double[] VERTICES = {0, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30};
  private static final double RHO_TENOR = 0.99;
  private static final double RHO_LOCATION = 0.999;
  private static final double GAMMA = 0.20;
  private static final double VEGA_LH = 120.0;
  private static final double VEGA_ALPHA = 0.01;

  public static final CommoditySbmParameters INSTANCE = new CommoditySbmParameters();

  private CommoditySbmParameters() {
  }

  @Override
  public RiskClass riskClass() {
    return RiskClass.COMMODITY;
  }

  @Override
  public double deltaRiskWeight(RiskFactor k) {
    return DELTA_RW[bucket(k)];
  }

  @Override
  public double vegaRiskWeight(RiskFactor k) {
    return Math.min(0.55 * Math.sqrt(VEGA_LH / 10.0), 1.0);
  }

  @Override
  public double curvatureShock(RiskFactor k, double riskFactorLevel) {
    return DELTA_RW[bucket(k)] * riskFactorLevel;
  }

  @Override
  public double deltaRho(RiskFactor k, RiskFactor l) {
    if (k.equals(l)) {
      return 1.0;
    }
    if (!k.bucket().equals(l.bucket())) {
      return 0.0;
    }
    double rc = k.commodityName().equals(l.commodityName()) ? 1.0 : RHO_COMMODITY[bucket(k)];
    double rt = k.tenor() == l.tenor() ? 1.0 : RHO_TENOR;
    double rl = k.deliveryLocation().equals(l.deliveryLocation()) ? 1.0 : RHO_LOCATION;
    return rc * rt * rl;
  }

  @Override
  public double vegaRho(RiskFactor k, RiskFactor l) {
    if (k.equals(l)) {
      return 1.0;
    }
    if (!k.bucket().equals(l.bucket())) {
      return 0.0;
    }
    double rc = k.commodityName().equals(l.commodityName()) ? 1.0 : RHO_COMMODITY[bucket(k)];
    double x = Math.max(k.tenor(), 1e-9);
    double y = Math.max(l.tenor(), 1e-9);
    double rOpt = Math.exp(-VEGA_ALPHA * Math.abs(x - y) / Math.min(x, y));
    return Math.min(rc * rOpt, 1.0);
  }

  @Override
  public double gamma(String bucketB, String bucketC) {
    return bucketB.equals(bucketC) ? 1.0 : GAMMA;
  }

  private static int bucket(RiskFactor k) {
    return Integer.parseInt(k.bucket());
  }

  /** The commodity maturity vertices in years. */
  public static double[] vertices() {
    return VERTICES.clone();
  }
}
