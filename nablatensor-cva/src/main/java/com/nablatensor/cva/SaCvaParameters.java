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
package com.nablatensor.cva;

import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskFactor;
import java.util.Map;

/**
 * Risk weights, correlations and the {@code m_CVA} multiplier for the SA-CVA
 * capital charge (Basel MAR50.9-50.53; CRR3 Art. 383). Every value here is
 * <em>indicative</em> — transcribed from the cited text for a runnable demo and
 * to be checked against the reader's current rulebook. The authoritative copy
 * belongs in the bank's parameter store, versioned, with provenance.
 *
 * @param mCva            the multiplier on the aggregated charge (supervisory floor 1.0)
 * @param creditSpreadRw  counterparty CS delta risk weight by SA-CVA bucket id
 * @param creditSpreadVegaRw counterparty CS vega risk weight
 * @param girrDeltaRw     interest-rate delta risk weight (per 1 bp sensitivity)
 * @param girrVegaRw      interest-rate vega risk weight
 * @param fxDeltaRw       FX delta risk weight
 * @param fxVegaRw        FX vega risk weight
 * @param creditSpreadRho within-bucket correlation between CS tenors of one name
 * @param creditSpreadGamma across-bucket correlation between CS buckets
 * @param fxGamma         across-bucket correlation between FX pairs
 */
public record SaCvaParameters(double mCva,
                              Map<String, Double> creditSpreadRw, double creditSpreadVegaRw,
                              double girrDeltaRw, double girrVegaRw,
                              double fxDeltaRw, double fxVegaRw,
                              double creditSpreadRho, double creditSpreadGamma, double fxGamma) {

  public SaCvaParameters {
    creditSpreadRw = Map.copyOf(creditSpreadRw);
    if (!(mCva >= 1.0)) {
      throw new IllegalArgumentException("m_CVA has a supervisory floor of 1.0, got " + mCva);
    }
  }

  /** MAR50 indicative demo tables. Not for regulatory use. */
  public static SaCvaParameters demo() {
    Map<String, Double> csRw = Map.of(
        "1", 0.005,   // sovereign IG
        "2", 0.010,   // local government
        "3", 0.050,   // financial
        "4", 0.030,   // corporate
        "5", 0.030,   // consumer
        "6", 0.020,   // tech / telecom
        "7", 0.120);  // other / high yield
    return new SaCvaParameters(1.0, csRw, 1.0,
        0.0111, 1.0,
        0.11, 1.0,
        0.90, 0.50, 0.60);
  }

  public double deltaRiskWeight(RiskFactor factor) {
    return switch (factor.riskClass()) {
      case GIRR -> girrDeltaRw;
      case CSR_NON_SEC -> creditSpreadRw.getOrDefault(factor.bucket(), 0.05);
      case FX -> fxDeltaRw;
      default -> throw new IllegalArgumentException("no SA-CVA delta RW for " + factor.riskClass());
    };
  }

  public double vegaRiskWeight(RiskFactor factor) {
    return switch (factor.riskClass()) {
      case GIRR -> girrVegaRw;
      case CSR_NON_SEC -> creditSpreadVegaRw;
      case FX -> fxVegaRw;
      default -> throw new IllegalArgumentException("no SA-CVA vega RW for " + factor.riskClass());
    };
  }

  public double withinBucketCorrelation(RiskClass riskClass) {
    return riskClass == RiskClass.CSR_NON_SEC ? creditSpreadRho : 0.99;
  }

  public double acrossBucketCorrelation(RiskClass riskClass) {
    return switch (riskClass) {
      case CSR_NON_SEC -> creditSpreadGamma;
      case FX -> fxGamma;
      default -> 0.0;
    };
  }
}
