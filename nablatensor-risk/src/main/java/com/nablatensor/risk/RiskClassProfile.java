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
package com.nablatensor.risk;

/**
 * The per-risk-class parameter surface the generic FRTB SA-SBM engine
 * ({@code com.nablatensor.reg.frtb.sbm.SbmCharge}) drives. One implementation
 * per risk class (GIRR, CSR non-sec / sec / CTP, equity, commodity, FX); the
 * implementation holds the MAR21 risk-weight, correlation and curvature-shock
 * tables and is the single place a value changes.
 *
 * <p>All correlations returned here are <b>MEDIUM</b>-scenario values; the
 * engine applies the {@link CorrelationScenario} HIGH / LOW transforms.
 *
 * <p>This interface carries no numbers. Its implementations do, and each of
 * those must cite the Basel paragraph and warn that the reader should verify
 * against the current rulebook — as {@code EquitySbmParameters} already does.
 */
public interface RiskClassProfile {

  /** The risk class this profile parameterises. */
  RiskClass riskClass();

  /** Delta risk weight for a delta risk factor (as a decimal fraction). */
  double deltaRiskWeight(RiskFactor k);

  /** Vega risk weight for a vega risk factor (as a decimal fraction). */
  double vegaRiskWeight(RiskFactor k);

  /**
   * The additive shock magnitude applied to {@code riskFactorLevel} for the
   * curvature up and down repricings. Relative-shock classes return
   * {@code rw * riskFactorLevel}; absolute-shock classes return {@code rw}.
   */
  double curvatureShock(RiskFactor k, double riskFactorLevel);

  /** Within-bucket delta correlation between two factors (MEDIUM scenario); {@code 1} if identical. */
  double deltaRho(RiskFactor k, RiskFactor l);

  /** Within-bucket vega correlation between two factors (MEDIUM scenario); {@code 1} if identical. */
  double vegaRho(RiskFactor k, RiskFactor l);

  /** Across-bucket correlation between two buckets (MEDIUM scenario); {@code 1} if the same bucket. */
  double gamma(String bucketB, String bucketC);
}
