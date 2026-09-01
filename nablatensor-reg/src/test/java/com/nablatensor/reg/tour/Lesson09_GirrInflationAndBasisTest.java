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

import com.nablatensor.reg.frtb.sbm.girr.GirrSbmParameters;
import com.nablatensor.risk.RiskFactor;
import org.junit.jupiter.api.Test;

/**
 * LESSON 9 — the two GIRR risk factors that are <b>not</b> tied to a tenor
 * vertex: <b>inflation</b> and <b>cross-currency basis</b> (MAR21).
 *
 * <pre>
 *   EUR bucket
 *   ┌─────────────────────────────────────────────┐
 *   │  OIS 0.25y ── 0.5y ── 1y ── ... ── 30y        │  the yield curve (lesson 7)
 *   │       \        |       /                       │
 *   │        \       |      /  rho = 40%             │
 *   │       [ INFLATION ]  ─────────────────────     │  one factor, flat vs the curve
 *   │                                                │
 *   │       [ XCCY BASIS ]  rho = 0% to everything   │  one factor, stands alone
 *   └─────────────────────────────────────────────┘
 * </pre>
 */
class Lesson09_GirrInflationAndBasisTest {

  private static final GirrSbmParameters GIRR = GirrSbmParameters.baselDefault();

  @Test
  void inflationCorrelatesWithTheYieldCurveAt40Percent() {
    double rho = GIRR.deltaRho(
        RiskFactor.girrInflation("EUR"), RiskFactor.girrDelta("EUR", "OIS", 5));
    assertEquals(0.40, rho, 1e-12);
  }

  @Test
  void crossCurrencyBasisIsUncorrelatedWithEverythingElse() {
    assertEquals(0.0, GIRR.deltaRho(
        RiskFactor.girrXccyBasis("EUR"), RiskFactor.girrDelta("EUR", "OIS", 5)), 1e-12);
    assertEquals(0.0, GIRR.deltaRho(
        RiskFactor.girrXccyBasis("EUR"), RiskFactor.girrInflation("EUR")), 1e-12);
  }

  @Test
  void inflationAndBasisStillGetARiskWeight() {
    // both carry a flat 1.6% weight (liquid-currency relief still applies to EUR)
    assertEquals(0.016 / Math.sqrt(2.0), GIRR.deltaRiskWeight(RiskFactor.girrInflation("EUR")), 1e-12);
    assertEquals(0.016 / Math.sqrt(2.0), GIRR.deltaRiskWeight(RiskFactor.girrXccyBasis("EUR")), 1e-12);
  }
}
