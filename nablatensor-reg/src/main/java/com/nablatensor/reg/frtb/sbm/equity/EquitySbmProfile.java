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
package com.nablatensor.reg.frtb.sbm.equity;

import com.nablatensor.reg.frtb.EquitySbmParameters;
import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskClassProfile;
import com.nablatensor.risk.RiskFactor;

/**
 * The equity risk class as a {@link RiskClassProfile}, so the generic
 * {@code SbmCharge} engine and the {@code FrtbSa} assembler treat equity like
 * every other class. Spot delta / vega / curvature delegate to the existing
 * {@link EquitySbmParameters} (published MAR21 equity values); this class adds
 * the <b>equity repo-rate</b> factor.
 *
 * <p><b>Repo-rate placeholder:</b> MAR21.78 gives equity repo-rate factors a
 * separate, smaller risk-weight table. Pending transcription this profile uses
 * the bucket's spot risk weight for repo factors (conservative) and
 * {@code rho(spot, repo of same issuer) = 0.999}. Replace with the MAR21.78
 * repo table before any real use.
 */
public final class EquitySbmProfile implements RiskClassProfile {

  private static final double SPOT_REPO_RHO = 0.999;

  public static final EquitySbmProfile INSTANCE = new EquitySbmProfile();

  private EquitySbmProfile() {
  }

  @Override
  public RiskClass riskClass() {
    return RiskClass.EQUITY;
  }

  @Override
  public double deltaRiskWeight(RiskFactor k) {
    return EquitySbmParameters.deltaRiskWeight(k);   // repo uses the spot weight (placeholder)
  }

  @Override
  public double vegaRiskWeight(RiskFactor k) {
    return EquitySbmParameters.vegaRiskWeight(k);
  }

  @Override
  public double curvatureShock(RiskFactor k, double riskFactorLevel) {
    // Relative spot shock; repo factors do not carry curvature (MAR21.98).
    return EquitySbmParameters.curvatureRiskWeight(Integer.parseInt(k.bucket())) * riskFactorLevel;
  }

  @Override
  public double deltaRho(RiskFactor k, RiskFactor l) {
    if (k.equals(l)) {
      return 1.0;
    }
    if (!k.bucket().equals(l.bucket())) {
      return 0.0;
    }
    if ((k.isEquityRepo() || l.isEquityRepo()) && k.name().equals(l.name())) {
      return SPOT_REPO_RHO;   // spot vs repo, or repo vs repo, of the same issuer
    }
    return EquitySbmParameters.deltaRho(spot(k), spot(l));
  }

  @Override
  public double vegaRho(RiskFactor k, RiskFactor l) {
    return EquitySbmParameters.vegaRho(k, l);
  }

  @Override
  public double gamma(String bucketB, String bucketC) {
    return EquitySbmParameters.gamma(bucketB, bucketC);
  }

  private static RiskFactor spot(RiskFactor f) {
    return RiskFactor.equityDelta(f.bucket(), f.name());
  }
}
