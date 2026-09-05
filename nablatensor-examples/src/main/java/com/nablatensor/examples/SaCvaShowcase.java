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
package com.nablatensor.examples;

import com.nablatensor.cva.CreditName;
import com.nablatensor.cva.CvaMarket;
import com.nablatensor.cva.CvaResult;
import com.nablatensor.cva.CvaRiskFactors;
import com.nablatensor.cva.ExposureSimulation;
import com.nablatensor.cva.FxForward;
import com.nablatensor.cva.HazardCurve;
import com.nablatensor.cva.InterestRateSwap;
import com.nablatensor.cva.NettingSet;
import com.nablatensor.cva.SaCva;
import com.nablatensor.cva.SaCvaParameters;
import com.nablatensor.cva.SaCvaResult;
import com.nablatensor.cva.SaCvaSensitivities;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The SA-CVA sensitivity vector for a small netting set, produced two ways:
 * one adjoint sweep, and the letter-compliant prescribed-bump revaluation the
 * adjoint sweep replaces. Both feed the same {@link SaCva} capital formula.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.SaCvaShowcase}
 */
public final class SaCvaShowcase {

  private static final long PATHS = 30_000L;
  private static final long SEED = 20260902L;

  private SaCvaShowcase() {
  }

  public static void main(String[] args) {
    CreditName counterparty = new CreditName("CPTY-A",
        HazardCurve.fromFlatSpread(150.0, 0.40, 10.0), 0.40,
        CreditName.Rating.BBB, CreditName.Sector.FINANCIAL);
    NettingSet nettingSet = new NettingSet("NS-CPTY-A", counterparty, List.of(
        InterestRateSwap.payer("SWAP-PAY", 100_000_000.0, 0.032, 5.0),
        InterestRateSwap.receiver("SWAP-REC", 40_000_000.0, 0.028, 5.0),
        new FxForward("FX-FWD", FxForward.Side.BUY_FOREIGN, 30_000_000.0, 1.10, 3.0)));
    CvaRiskFactors keys = new CvaRiskFactors("USD", nettingSet.counterparty(), "EURUSD");

    System.out.printf(Locale.ROOT,
        "Netting set NS-CPTY-A: 2 interest-rate swaps + 1 FX forward, one BBB counterparty%n");
    System.out.printf(Locale.ROOT, "%,d exposure paths, seed %d, engine cpu-jit%n%n", PATHS, SEED);

    ExposureSimulation simulation = new ExposureSimulation(nettingSet, 20).on("cpu-jit");
    CvaMarket base = CvaMarket.demo();

    CvaResult swept = simulation.run(base, PATHS, SEED);
    System.out.printf(Locale.ROOT, "unilateral CVA           %,.2f  (se %.2f)%n",
        swept.value(), swept.standardError());
    System.out.printf(Locale.ROOT, "adjoint sweep            %.3f s  (value + full CvaMarket gradient)%n%n",
        swept.sweepSeconds());

    Sensitivities adjoint = SaCvaSensitivities.adjoint(swept, keys);
    SaCvaSensitivities.BumpResult bump =
        SaCvaSensitivities.bumpAndRevalue(simulation, base, PATHS, SEED, keys);

    System.out.printf(Locale.ROOT, "%-32s %14s %14s%n", "risk factor", "adjoint", "bump");
    for (Map.Entry<RiskFactor, Double> entry : bump.sensitivities().asMap().entrySet()) {
      double bumped = entry.getValue();
      double swept1 = adjoint.get(entry.getKey());
      System.out.printf(Locale.ROOT, "%-32s %14.4f %14.4f%n", entry.getKey(), swept1, bumped);
    }
    System.out.printf(Locale.ROOT, "%nbump-and-revalue         %d netting-set re-simulations, %.3f s%n",
        bump.revaluations(), bump.seconds());
    System.out.printf(Locale.ROOT, "speedup                  %.1fx%n%n",
        bump.seconds() / swept.sweepSeconds());

    SaCvaResult fromAdjoint = new SaCva(SaCvaParameters.demo()).charge(adjoint);
    SaCvaResult fromBump = new SaCva(SaCvaParameters.demo()).charge(bump.sensitivities());
    System.out.printf(Locale.ROOT, "SA-CVA charge  from adjoint sweep   %,.2f  (scenario %s)%n",
        fromAdjoint.total(), fromAdjoint.selected());
    System.out.printf(Locale.ROOT, "SA-CVA charge  from prescribed bump %,.2f  (scenario %s)%n",
        fromBump.total(), fromBump.selected());
  }
}
