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

/**
 * A CVA credit hedge recognised by BA-CVA (full version, MAR50.6) and SA-CVA.
 * A single-name CDS references one counterparty and gets a correlation
 * {@code r_hc} to it (1.0 for the same legal entity, ~0.8 for a legally related
 * name, ~0.5 for a same-sector proxy); an index CDS hedges the portfolio and is
 * recognised through the {@code IH} term.
 *
 * @param kind          single-name or index CDS
 * @param referenceId   the counterparty id a single-name CDS hedges ({@code ""} for an index)
 * @param notional      hedge notional in reporting currency
 * @param maturityYears remaining maturity in years
 * @param riskWeight    the hedge's supervisory credit risk weight (RW_h)
 * @param correlation   {@code r_hc} to the hedged counterparty (single-name only; {@code 1.0} for an index)
 */
public record CvaHedge(Kind kind, String referenceId, double notional, double maturityYears,
                       double riskWeight, double correlation) {

  public enum Kind { SINGLE_NAME_CDS, INDEX_CDS }

  public CvaHedge {
    if (!(notional >= 0.0) || !(maturityYears > 0.0) || !(riskWeight >= 0.0)) {
      throw new IllegalArgumentException("need notional>=0, maturityYears>0, riskWeight>=0");
    }
    if (!(correlation >= 0.0 && correlation <= 1.0)) {
      throw new IllegalArgumentException("r_hc must be in [0, 1], got " + correlation);
    }
  }

  public static CvaHedge singleName(String counterpartyId, double notional, double maturityYears,
                                    double riskWeight, double correlation) {
    return new CvaHedge(Kind.SINGLE_NAME_CDS, counterpartyId, notional, maturityYears,
        riskWeight, correlation);
  }

  public static CvaHedge index(double notional, double maturityYears, double riskWeight) {
    return new CvaHedge(Kind.INDEX_CDS, "", notional, maturityYears, riskWeight, 1.0);
  }

  /** Supervisory discount factor {@code (1 - e^{-0.05 M}) / (0.05 M)}. */
  public double supervisoryDiscount() {
    double x = 0.05 * maturityYears;
    return x <= 0.0 ? 1.0 : (1.0 - Math.exp(-x)) / x;
  }

  /** {@code RW_h * M_h * B_h * DF_h} — the hedge's contribution magnitude. */
  public double discountedWeightedNotional() {
    return riskWeight * maturityYears * notional * supervisoryDiscount();
  }
}
