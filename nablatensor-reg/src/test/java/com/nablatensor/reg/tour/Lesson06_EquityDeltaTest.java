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

import com.nablatensor.reg.frtb.EquitySbmParameters;
import com.nablatensor.reg.frtb.sbm.SbmCharge;
import com.nablatensor.reg.frtb.sbm.equity.EquitySbmProfile;
import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * LESSON 6 — the first <b>real</b> risk class: equity delta (MAR21.12), with the
 * published risk weights and correlations from {@link EquitySbmParameters}.
 *
 * <p>Equity delta risk factors are the <b>spot prices</b> of individual names /
 * indices, grouped into 13 buckets by market cap, economy and sector. The book
 * sensitivity {@code s_k = dPV/dSpot_k} is one column of the adjoint Jacobian.
 *
 * <pre>
 *   bucket 5 (large cap, advanced economy, consumer)
 *   risk weight 30% ; two distinct names correlate at rho = 25%
 *
 *   s(ACME)   = +1000      WS(ACME)   = 0.30 *  1000 = +300
 *   s(GLOBEX) =  -400      WS(GLOBEX) = 0.30 * (-400) = -120
 *
 *   K_b = sqrt( 300^2 + 120^2 + 2 * 0.25 * 300 * (-120) )
 *       = sqrt( 90000 + 14400 - 18000 ) = sqrt(86400) ~ 293.94
 * </pre>
 */
class Lesson06_EquityDeltaTest {

  @Test
  void publishedParameters_areWhatWeExpect() {
    RiskFactor acme = RiskFactor.equityDelta("5", "ACME");
    assertEquals(0.30, EquitySbmParameters.deltaRiskWeight(acme), 1e-12);
    // buckets 1-4 correlate at 15%, buckets 5-8 at 25%
    assertEquals(0.25, EquitySbmParameters.deltaRho(
        RiskFactor.equityDelta("5", "ACME"), RiskFactor.equityDelta("5", "GLOBEX")), 1e-12);
    assertEquals(0.15, EquitySbmParameters.deltaRho(
        RiskFactor.equityDelta("1", "A"), RiskFactor.equityDelta("1", "B")), 1e-12);
  }

  @Test
  void oneBucketTwoNames_matchesTheHandCalc() {
    Sensitivities book = Sensitivities.builder()
        .add(RiskFactor.equityDelta("5", "ACME"), 1000.0)
        .add(RiskFactor.equityDelta("5", "GLOBEX"), -400.0)
        .build();

    double medium = SbmCharge.of(EquitySbmProfile.INSTANCE)
        .compute(book, List.of())
        .perScenario().get(CorrelationScenario.MEDIUM);

    assertEquals(Math.sqrt(86_400.0), medium, 1e-6);   // ~293.94
  }

  @Test
  void longShortBook_isBoundByTheLowScenario() {
    Sensitivities book = Sensitivities.builder()
        .add(RiskFactor.equityDelta("5", "ACME"), 1000.0)
        .add(RiskFactor.equityDelta("5", "GLOBEX"), -400.0)
        .build();

    SbmCharge.Result r = SbmCharge.of(EquitySbmProfile.INSTANCE).compute(book, List.of());
    assertEquals(CorrelationScenario.LOW, r.bindingScenario());
  }
}
