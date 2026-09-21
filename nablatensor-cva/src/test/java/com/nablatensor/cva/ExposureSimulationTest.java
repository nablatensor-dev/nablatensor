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

import com.nablatensor.risk.TimeProfile;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("mc")
class ExposureSimulationTest {

  private static final long PATHS = 60_000L;
  private static final long SEED = 20260902L;

  private static NettingSet demoNettingSet() {
    CreditName counterparty = CreditName.of().id("CPTY-A").curve(HazardCurve.fromFlatSpread(150.0, 0.40, 10.0)).recovery(0.40).rating(CreditName.RatingEnum.BBB).sector(CreditName.SectorEnum.FINANCIAL).build();
    return NettingSet.of().id("NS-CPTY-A").counterparty(counterparty).trades(List.of(
        InterestRateSwap.of().id("SWAP-PAY").side(InterestRateSwap.SideEnum.PAY_FIXED).notional(100_000_000.0).fixedRate(0.032).startYears(0.0).maturityYears(5.0).accrualYears(0.5).build(),
        InterestRateSwap.of().id("SWAP-REC").side(InterestRateSwap.SideEnum.RECEIVE_FIXED).notional(40_000_000.0).fixedRate(0.028).startYears(0.0).maturityYears(5.0).accrualYears(0.5).build())).collateral(CollateralAgreement.uncollateralised()).build();
  }

  @Test
  void cvaIsPositiveAndExposureProfileIsHumped() {
    ExposureSimulation simulation = ExposureSimulation.of(demoNettingSet(), 20).on("cpu-jit");
    CvaResult result = simulation.run(CvaMarket.demo(), PATHS, SEED);

    assertTrue(result.value() > 0.0, "CVA must be a positive charge, got " + result.value());

    TimeProfile epe = result.epeProfile();
    double first = epe.values()[0];
    double last = epe.values()[epe.values().length - 1];
    double peak = result.peakExpectedExposure();
    assertTrue(peak > first && peak > last,
        "expected a mid-life exposure hump: first=" + first + " peak=" + peak + " last=" + last);
    assertTrue(last < 0.25 * peak, "exposure should amortise toward maturity");
  }

  @Test
  void collateralReducesTheCharge() {
    NettingSet uncollateralised = demoNettingSet();
    NettingSet margined = NettingSet.of().id(uncollateralised.id()).counterparty(uncollateralised.counterparty()).trades(uncollateralised.trades()).collateral(CollateralAgreement.dailyMargined(0.0)).build();

    double withoutCsa = ExposureSimulation.of(uncollateralised, 20).on("cpu-jit")
        .run(CvaMarket.demo(), PATHS, SEED).value();
    double withCsa = ExposureSimulation.of(margined, 20).on("cpu-jit")
        .run(CvaMarket.demo(), PATHS, SEED).value();

    assertTrue(withCsa < withoutCsa,
        "a daily-margined CSA must cut the CVA: " + withCsa + " vs " + withoutCsa);
    assertTrue(withCsa >= 0.0, "collateralised CVA still non-negative");
  }

  @Test
  void oneSweepGradientMatchesBumpAndRevalue() {
    ExposureSimulation simulation = ExposureSimulation.of(demoNettingSet(), 20).on("cpu-jit");
    CvaMarket base = CvaMarket.demo();
    CvaResult swept = simulation.run(base, PATHS, SEED);

    double h = 1.0e-4;
    double bumpRate = (simulation.cvaOnly(base.withShortRate(base.r0() + h), PATHS, SEED)
        - simulation.cvaOnly(base.withShortRate(base.r0() - h), PATHS, SEED)) / (2.0 * h);
    assertEquals(bumpRate, swept.gradient().r0(),
        1.0e-2 * (1.0 + Math.abs(bumpRate)) + 5.0,
        "dCVA/dr0: one adjoint sweep vs central bump");

    double hHazard = 1.0e-5;
    double bumpHazard = (simulation.cvaOnly(base.withHazardParallelShift(hHazard), PATHS, SEED)
        - simulation.cvaOnly(base.withHazardParallelShift(-hHazard), PATHS, SEED)) / (2.0 * hHazard);
    double adjointHazard = swept.gradient().hazardShort()
        + swept.gradient().hazardMid() + swept.gradient().hazardLong();
    assertEquals(bumpHazard, adjointHazard,
        2.0e-2 * (1.0 + Math.abs(bumpHazard)) + 5.0,
        "CS01 (sum of hazard-bucket sensitivities): one sweep vs parallel bump");
    assertTrue(adjointHazard > 0.0, "wider spreads raise the CVA");
  }
}
