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
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * LESSON 13 — vega risk. A separate charge, same machinery: the risk factor is
 * an option's <b>implied volatility</b>, indexed by option maturity, and it uses
 * the <b>same buckets</b> as delta.
 *
 * <pre>
 *   vega risk weight = min( 0.55 * sqrt(LH / 10) , 1 )
 *
 *   LH  = liquidity horizon of the risk class:
 *         equity large cap (buckets 1-8, 12) : LH = 20   -> RW = 0.55*sqrt(2)  ~ 0.778
 *         equity small cap / other           : LH = 120  -> RW = min(1.905, 1) = 1.000
 * </pre>
 */
class Lesson13_VegaRiskTest {

  @Test
  void vegaRiskWeightDependsOnTheLiquidityHorizon() {
    double largeCap = EquitySbmParameters.vegaRiskWeight(RiskFactor.equityVega("5", "ACME", 1));
    double smallCap = EquitySbmParameters.vegaRiskWeight(RiskFactor.equityVega("9", "TINYCO", 1));

    assertEquals(0.55 * Math.sqrt(2.0), largeCap, 1e-12);
    assertEquals(1.0, smallCap, 1e-12);
  }

  @Test
  void vegaCorrelationDecaysWithOptionMaturityDistance() {
    // same name, same bucket: rho = (name correlation = 1) * exp(-0.01 * |t1 - t2| / min(t1, t2))
    double rho = EquitySbmParameters.vegaRho(
        RiskFactor.equityVega("5", "ACME", 1), RiskFactor.equityVega("5", "ACME", 2));
    assertEquals(Math.exp(-0.01), rho, 1e-9);
  }

  @Test
  void aVegaOnlyBookProducesAVegaCharge() {
    Sensitivities book = Sensitivities.builder()
        .add(RiskFactor.equityVega("5", "ACME", 1), 200.0)
        .add(RiskFactor.equityVega("5", "ACME", 5), -60.0)
        .build();

    SbmCharge.Result r = SbmCharge.of(EquitySbmProfile.INSTANCE).compute(book, List.of());
    assertEquals(0.0, r.delta(), 0.0);
    assertEquals(0.0, r.curvature(), 0.0);
    assertEquals(r.vega(), r.total(), 1e-12, "for a vega-only book the class charge is entirely vega");
  }
}
