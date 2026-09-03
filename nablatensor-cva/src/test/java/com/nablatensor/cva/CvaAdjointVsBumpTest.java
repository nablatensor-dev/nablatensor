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

import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * The Phase-2 benchmark artefact: the full SA-CVA sensitivity vector from one
 * adjoint sweep must reconcile, factor by factor, with the letter-compliant
 * prescribed-bump vector — which costs one full netting-set exposure
 * re-simulation per shocked risk factor.
 */
class CvaAdjointVsBumpTest {

  private static final long PATHS = 30_000L;
  private static final long SEED = 20260902L;

  private static NettingSet mixedNettingSet() {
    CreditName counterparty = new CreditName("CPTY-A",
        HazardCurve.fromFlatSpread(150.0, 0.40, 10.0), 0.40,
        CreditName.Rating.BBB, CreditName.Sector.FINANCIAL);
    return new NettingSet("NS-CPTY-A", counterparty, List.of(
        InterestRateSwap.payer("SWAP-PAY", 100_000_000.0, 0.032, 5.0),
        InterestRateSwap.receiver("SWAP-REC", 40_000_000.0, 0.028, 5.0),
        new FxForward("FX-FWD", FxForward.Side.BUY_FOREIGN, 30_000_000.0, 1.10, 3.0)));
  }

  @Test
  void oneSweepReconcilesWithPrescribedBumpVector() {
    NettingSet nettingSet = mixedNettingSet();
    CvaRiskFactors keys = new CvaRiskFactors("USD", nettingSet.counterparty(), "EURUSD");
    ExposureSimulation simulation = new ExposureSimulation(nettingSet, 20).on("cpu-jit");
    CvaMarket base = CvaMarket.demo();

    CvaResult swept = simulation.run(base, PATHS, SEED);
    Sensitivities adjoint = SaCvaSensitivities.adjoint(swept, keys);
    SaCvaSensitivities.BumpResult bump =
        SaCvaSensitivities.bumpAndRevalue(simulation, base, PATHS, SEED, keys);

    assertTrue(bump.revaluations() >= 14,
        "the prescribed-bump vector re-simulates the whole netting set per factor");
    assertEquals(adjoint.asMap().size(), bump.sensitivities().asMap().size(),
        "same set of risk factors both ways");

    for (Map.Entry<RiskFactor, Double> entry : bump.sensitivities().asMap().entrySet()) {
      double bumped = entry.getValue();
      double swept1 = adjoint.get(entry.getKey());
      assertEquals(bumped, swept1,
          3.0e-2 * (1.0 + Math.abs(bumped)) + 2.0,
          "adjoint vs bump for " + entry.getKey());
    }

    // both charges agree once aggregated
    SaCvaResult fromAdjoint = new SaCva(SaCvaParameters.demo()).charge(adjoint);
    SaCvaResult fromBump = new SaCva(SaCvaParameters.demo()).charge(bump.sensitivities());
    assertEquals(fromBump.total(), fromAdjoint.total(),
        3.0e-2 * (1.0 + Math.abs(fromBump.total())) + 2.0,
        "SA-CVA charge: one sweep vs " + bump.revaluations() + " re-simulations");
  }
}
