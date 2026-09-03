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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import org.junit.jupiter.api.Test;

class SaCvaTest {

  private static final SaCvaParameters PARAMETERS = SaCvaParameters.demo();

  @Test
  void chargeIsNonNegativeCoversThreeScenariosAndPicksTheMax() {
    Sensitivities sensitivities = Sensitivities.builder()
        .add(RiskFactor.girrDelta("USD", 5.0), 40_000.0)
        .add(RiskFactor.csrDelta("3", "CPTY-A", RiskFactor.CsrCurve.CDS, 1.0), 12_000.0)
        .add(RiskFactor.csrDelta("3", "CPTY-A", RiskFactor.CsrCurve.CDS, 3.5), 9_000.0)
        .add(RiskFactor.fxDelta("EURUSD"), 5_000.0)
        .build();

    SaCvaResult result = new SaCva(PARAMETERS).charge(sensitivities);

    assertEquals(CorrelationScenario.values().length, result.perScenario().size());
    for (double charge : result.perScenario().values()) {
      assertTrue(charge >= 0.0, "each scenario charge is non-negative");
    }
    double max = result.perScenario().values().stream().mapToDouble(Double::doubleValue).max().orElseThrow();
    assertEquals(PARAMETERS.mCva() * max, result.total(), 1.0e-9);
    assertEquals(max, result.perScenario().get(result.selected()), 1.0e-9);
  }

  @Test
  void singleFactorChargeIsRiskWeightTimesSensitivity() {
    RiskFactor factor = RiskFactor.girrDelta("USD", 5.0);
    Sensitivities one = Sensitivities.builder().add(factor, 25_000.0).build();

    SaCvaResult result = new SaCva(PARAMETERS).charge(one);

    // one risk type, one bucket, one factor: K = m_CVA * |RW * s|
    double expected = PARAMETERS.mCva() * PARAMETERS.deltaRiskWeight(factor) * 25_000.0;
    assertEquals(expected, result.total(), 1.0e-6 * expected);
    assertEquals(expected, result.byRiskType().get(RiskClass.GIRR), 1.0e-6 * expected);
  }

  @Test
  void mCvaMultiplierScalesTheChargeLinearly() {
    Sensitivities sensitivities = Sensitivities.builder()
        .add(RiskFactor.girrDelta("USD", 5.0), 40_000.0)
        .add(RiskFactor.fxDelta("EURUSD"), 8_000.0)
        .build();

    double atOne = new SaCva(PARAMETERS).charge(sensitivities).total();
    SaCvaParameters raised = new SaCvaParameters(1.5, PARAMETERS.creditSpreadRw(),
        PARAMETERS.creditSpreadVegaRw(), PARAMETERS.girrDeltaRw(), PARAMETERS.girrVegaRw(),
        PARAMETERS.fxDeltaRw(), PARAMETERS.fxVegaRw(), PARAMETERS.creditSpreadRho(),
        PARAMETERS.creditSpreadGamma(), PARAMETERS.fxGamma());
    double atOnePointFive = new SaCva(raised).charge(sensitivities).total();

    assertEquals(1.5, atOnePointFive / atOne, 1.0e-9);
  }
}
