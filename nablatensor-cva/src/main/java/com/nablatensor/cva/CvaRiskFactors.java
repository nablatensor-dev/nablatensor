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

/**
 * The regulatory keys the CVA sensitivities of one netting set map onto: the
 * reporting-currency interest-rate curve, the counterparty credit-spread bucket,
 * and the FX pair. Used by {@link SaCvaSensitivities} to turn a {@link CvaMarket}
 * gradient into a bucketed {@code Sensitivities} vector.
 *
 * @param currency     reporting-currency GIRR curve id (e.g. {@code "USD"})
 * @param counterparty the netting-set counterparty (its sector sets the CS bucket)
 * @param fxPair       the FX pair the netting set carries (e.g. {@code "EURUSD"})
 */
public record CvaRiskFactors(String currency, CreditName counterparty, String fxPair) {

  private static final double IR_VERTEX_YEARS = 5.0;
  private static final double[] CS_VERTEX_YEARS = {1.0, 3.5, 7.5};

  public RiskFactor irDelta() {
    return RiskFactor.girrDelta(currency, IR_VERTEX_YEARS);
  }

  public RiskFactor irVega() {
    return RiskFactor.girrVega(currency, IR_VERTEX_YEARS, IR_VERTEX_YEARS);
  }

  public RiskFactor counterpartySpreadDelta(int bucketVertex) {
    return RiskFactor.csrDelta(counterpartyBucket(), counterparty.id(),
        RiskFactor.CsrCurve.CDS, CS_VERTEX_YEARS[bucketVertex]);
  }

  public RiskFactor fxDelta() {
    return RiskFactor.fxDelta(fxPair);
  }

  public RiskFactor fxVega() {
    return RiskFactor.fxVega(fxPair, IR_VERTEX_YEARS);
  }

  /** SA-CVA counterparty credit-spread bucket id from the counterparty sector (MAR50.10, indicative). */
  public String counterpartyBucket() {
    return switch (counterparty.sector()) {
      case SOVEREIGN -> "1";
      case LOCAL_GOVERNMENT -> "2";
      case FINANCIAL -> "3";
      case CORPORATE -> "4";
      case CONSUMER -> "5";
      case TECH -> "6";
      case OTHER -> "7";
    };
  }

  public static int creditSpreadVertexCount() {
    return CS_VERTEX_YEARS.length;
  }

  static RiskClass irRiskClass() {
    return RiskClass.GIRR;
  }
}
