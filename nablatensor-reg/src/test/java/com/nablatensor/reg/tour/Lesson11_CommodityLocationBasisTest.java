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
import com.nablatensor.reg.frtb.sbm.commodity.CommoditySbmParameters;
import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * LESSON 11 — commodity risk: the same "product of correlations" idea as CSR
 * (lesson 10), but the three dimensions are <b>commodity, tenor and delivery
 * location</b> (MAR21):
 *
 * <pre>
 *   rho = rho_commodity  *  rho_tenor  *  rho_location
 *
 *   WTI @ Cushing  vs  WTI @ Houston, both 1y:
 *       rho_commodity = 1     (same commodity)
 *       rho_tenor     = 1     (same maturity)
 *       rho_location  = 0.999 (different delivery point)
 *   -> a location spread ("Cushing vs Houston") is 99.9% hedged, 0.1% residual
 * </pre>
 */
class Lesson11_CommodityLocationBasisTest {

  private static final CommoditySbmParameters CTY = CommoditySbmParameters.INSTANCE;

  @Test
  void sameCommoditySameTenor_differentLocation_correlateAt0_999() {
    double rho = CTY.deltaRho(
        RiskFactor.commodityDelta("2", "WTI", 1, "CUSHING"),
        RiskFactor.commodityDelta("2", "WTI", 1, "HOUSTON"));
    assertEquals(0.999, rho, 1e-12);
  }

  @Test
  void differentCommoditiesInABucket_useThePerBucketCommodityCorrelation() {
    // bucket 2 (liquid energy) : distinct commodities correlate at 0.95
    double rho = CTY.deltaRho(
        RiskFactor.commodityDelta("2", "WTI", 1, "CUSHING"),
        RiskFactor.commodityDelta("2", "BRENT", 1, "CUSHING"));
    assertEquals(0.95, rho, 1e-12);
  }

  @Test
  void aLocationSpread_leavesASmallResidualCharge() {
    Sensitivities book = Sensitivities.builder()
        .add(RiskFactor.commodityDelta("2", "WTI", 1, "CUSHING"), 100.0 / 0.35)   // WS = +100 (rw 35%)
        .add(RiskFactor.commodityDelta("2", "WTI", 1, "HOUSTON"), -100.0 / 0.35)  // WS = -100
        .build();

    double medium = SbmCharge.of(CTY).compute(book, List.of())
        .perScenario().get(CorrelationScenario.MEDIUM);

    assertEquals(Math.sqrt(20_000.0 * (1 - 0.999)), medium, 1e-6);   // sqrt(20)
  }
}
