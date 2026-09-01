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
package com.nablatensor.reg.tour;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.nablatensor.reg.frtb.sbm.SbmCharge;
import com.nablatensor.reg.frtb.sbm.fx.FxSbmParameters;
import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * LESSON 12 — FX risk. The risk factors are exchange rates <b>relative to the
 * bank's reporting currency</b> (MAR21):
 *
 * <pre>
 *   reporting currency = EUR
 *
 *   bucket "EURUSD" : one factor, rate EUR/USD, RW 15% (/sqrt(2) — liquid pair)
 *   bucket "EURJPY" : one factor, rate EUR/JPY, RW 15% (/sqrt(2))
 *   bucket "EURPLN" : one factor, rate EUR/PLN, RW 15%      (not a liquid pair)
 *
 *   one factor per bucket  ->  within-bucket rho is trivial (1)
 *   across buckets          ->  gamma = 60%
 * </pre>
 */
class Lesson12_FxRiskAndReportingCurrencyTest {

  private static final FxSbmParameters FX = FxSbmParameters.INSTANCE;

  @Test
  void baseRiskWeightIs15Percent_withSqrt2ReliefForLiquidPairs() {
    assertEquals(0.15 / Math.sqrt(2.0), FX.deltaRiskWeight(RiskFactor.fxDelta("EURUSD")), 1e-12);
    assertEquals(0.15, FX.deltaRiskWeight(RiskFactor.fxDelta("EURPLN")), 1e-12);
  }

  @Test
  void oneFactorPerBucket_soWithinBucketCorrelationIsTrivial() {
    assertEquals(1.0, FX.deltaRho(RiskFactor.fxDelta("EURUSD"), RiskFactor.fxDelta("EURUSD")), 1e-12);
    assertEquals(0.0, FX.deltaRho(RiskFactor.fxDelta("EURUSD"), RiskFactor.fxDelta("EURJPY")), 1e-12);
  }

  @Test
  void currencyPairsDiversifyAcrossBucketsAt60Percent() {
    assertEquals(0.60, FX.gamma("EURUSD", "EURJPY"), 1e-12);

    Sensitivities book = Sensitivities.builder()
        .add(RiskFactor.fxDelta("EURUSD"), 1_000_000.0)
        .add(RiskFactor.fxDelta("EURGBP"), 1_000_000.0)
        .build();
    double medium = SbmCharge.of(FX).compute(book, List.of())
        .perScenario().get(CorrelationScenario.MEDIUM);

    // WS per bucket = 0.15/sqrt(2) * 1e6 ; charge = WS * sqrt(1 + 1 + 2*0.6)
    double ws = 0.15 / Math.sqrt(2.0) * 1_000_000.0;
    assertEquals(ws * Math.sqrt(3.2), medium, 1e-3);
  }
}
