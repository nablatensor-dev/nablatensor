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

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("mc")
class CvaAssemblerTest {

  private static final long PATHS = 40_000L;

  private static NettingSet swapNettingSet(String id, String counterpartyId,
                                           CreditName.SectorEnum sector) {
    CreditName counterparty = CreditName.of().id(counterpartyId).curve(HazardCurve.fromFlatSpread(150.0, 0.40, 10.0)).recovery(0.40).rating(CreditName.RatingEnum.BBB).sector(sector).build();
    return NettingSet.of().id(id).counterparty(counterparty).trades(List.of(
        InterestRateSwap.of().id("SWAP-PAY").side(InterestRateSwap.SideEnum.PAY_FIXED).notional(120_000_000.0).fixedRate(0.032).startYears(0.0).maturityYears(5.0).accrualYears(0.5).build(),
        InterestRateSwap.of().id("SWAP-REC").side(InterestRateSwap.SideEnum.RECEIVE_FIXED).notional(45_000_000.0).fixedRate(0.028).startYears(0.0).maturityYears(4.0).accrualYears(0.5).build())).collateral(CollateralAgreement.uncollateralised()).build();
  }

  @Test
  void portfolioRunProducesBothChargesAndNettedSensitivities() {
    NettingSet nsA = swapNettingSet("NS-A", "CPTY-A", CreditName.SectorEnum.FINANCIAL);
    NettingSet nsB = swapNettingSet("NS-B", "CPTY-B", CreditName.SectorEnum.CORPORATE);

    CvaCapital capital = Cva.of(CvaMarket.demo())
        .add(nsA, new CvaRiskFactors("USD", nsA.counterparty(), "EURUSD"))
        .add(nsB, new CvaRiskFactors("USD", nsB.counterparty(), "EURUSD"))
        .steps(20).paths(PATHS).on("cpu-jit")
        .compute();

    assertEquals(2, capital.perNettingSet().size());
    assertTrue(capital.cvaValue() > 0.0, "portfolio CVA is a positive charge");
    assertTrue(capital.baCva().reduced() > 0.0, "BA-CVA reduced is positive");
    assertTrue(capital.baCva().full() > 0.0, "BA-CVA full is positive");
    assertTrue(capital.saCva().total() > 0.0, "SA-CVA total is positive");
    assertEquals(2, capital.scvaByCounterparty().size());

    // netted vector carries IR delta and per-name CS01
    assertTrue(capital.aggregateSensitivities().asMap().size() >= 5);
    assertTrue(capital.bumpRevaluations() >= 2 * 8 * 2, "counts the re-simulations the sweep avoids");
  }

  @Test
  void praThreeMethodsAreAllAvailableAndOrdered() {
    NettingSet ns = swapNettingSet("NS-A", "CPTY-A", CreditName.SectorEnum.FINANCIAL);
    CvaCapital capital = Cva.of(CvaMarket.demo())
        .add(ns, new CvaRiskFactors("USD", ns.counterparty(), "EURUSD"))
        .steps(20).paths(PATHS).on("cpu-jit")
        .compute();

    PraCvaMethods methods = PraCvaMethods.of(capital);
    assertTrue(methods.alternativeApproach() > 0.0, "PRA Alternative Approach available");
    assertTrue(methods.basicApproach() > 0.0, "PRA Basic Approach (BA-CVA) available");
    assertTrue(methods.standardisedApproach() > 0.0, "PRA Standardised Approach (SA-CVA) available");
    assertEquals(
        Math.max(methods.alternativeApproach(),
            Math.max(methods.basicApproach(), methods.standardisedApproach())),
        methods.bindingCharge(), 1.0e-9);
  }
}
