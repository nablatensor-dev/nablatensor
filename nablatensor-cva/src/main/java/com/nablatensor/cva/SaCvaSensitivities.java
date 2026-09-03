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

import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;

/**
 * The SA-CVA sensitivity vector for a netting set, produced two ways:
 *
 * <ul>
 *   <li><b>Route B — {@link #adjoint}</b>: read straight off the one adjoint
 *       sweep in {@link CvaResult#gradient()}. {@code O(1)} regardless of how
 *       many risk factors the netting set touches.</li>
 *   <li><b>Route A — {@link #bumpAndRevalue}</b>: the letter-compliant
 *       prescribed bump — re-simulate the whole netting-set exposure once per
 *       shocked risk factor and central-difference. {@code O(#risk factors)}
 *       re-simulations; this is the cost the adjoint sweep removes.</li>
 * </ul>
 *
 * <p>Both return a {@code Sensitivities} keyed by {@link RiskFactor}, fed
 * straight into {@link SaCva}. On common random numbers the two agree to the
 * bump's own {@code O(h^2)} error — the reconciliation the benchmark rests on.
 */
public final class SaCvaSensitivities {

  /** One-basis-point move for the rates and credit-spread deltas. */
  public static final double BP = 1.0e-4;
  /** Relative move for the vega and FX deltas (1%). */
  public static final double RELATIVE = 0.01;

  private SaCvaSensitivities() {
  }

  /** Route A's outcome: the vector plus the price of getting it. */
  public record BumpResult(Sensitivities sensitivities, int revaluations, double seconds) {}

  // ---- Route B: one adjoint sweep -----------------------------------

  /**
   * The SA-CVA sensitivities read straight off the adjoint gradient. Rate and
   * credit-spread deltas are scaled to a 1 bp move; the credit-spread grad is
   * converted from hazard to par-spread by {@code d_lambda / d_s = 1 / (1 - R)};
   * the rate and FX vegas follow the FRTB convention {@code dCVA/dsigma * sigma}.
   */
  public static Sensitivities adjoint(CvaResult swept, CvaRiskFactors keys) {
    CvaMarket gradient = swept.gradient();
    CvaMarket market = swept.market();
    double lgd = keys.counterparty().lossGivenDefault();

    Sensitivities.Builder out = Sensitivities.builder();
    out.add(keys.irDelta(), (gradient.r0() + gradient.hwLevel()) * BP);
    out.add(keys.irVega(), gradient.hwSigma() * market.hwSigma());

    double[] hazardGradient = {gradient.hazardShort(), gradient.hazardMid(), gradient.hazardLong()};
    for (int vertex = 0; vertex < CvaRiskFactors.creditSpreadVertexCount(); vertex++) {
      out.add(keys.counterpartySpreadDelta(vertex), hazardGradient[vertex] / lgd * BP);
    }

    out.add(keys.fxDelta(), gradient.fxSpot() * market.fxSpot() * RELATIVE);
    out.add(keys.fxVega(), gradient.fxVol() * market.fxVol());
    return out.build();
  }

  // ---- Route A: prescribed bump & re-simulate ----------------------

  public static BumpResult bumpAndRevalue(ExposureSimulation simulation, CvaMarket base,
                                          long paths, long seed, CvaRiskFactors keys) {
    long start = System.nanoTime();
    int revaluations = 0;
    double lgd = keys.counterparty().lossGivenDefault();
    Sensitivities.Builder out = Sensitivities.builder();

    double irUp = simulation.cvaOnly(base.withCurveLevel(base.r0() + BP, base.hwLevel() + BP), paths, seed);
    double irDown = simulation.cvaOnly(base.withCurveLevel(base.r0() - BP, base.hwLevel() - BP), paths, seed);
    revaluations += 2;
    out.add(keys.irDelta(), 0.5 * (irUp - irDown));

    double sig = base.hwSigma();
    double vegaUp = simulation.cvaOnly(base.withRateVol(sig * (1.0 + RELATIVE)), paths, seed);
    double vegaDown = simulation.cvaOnly(base.withRateVol(sig * (1.0 - RELATIVE)), paths, seed);
    revaluations += 2;
    out.add(keys.irVega(), (vegaUp - vegaDown) / (2.0 * RELATIVE));

    double[] hazard = {base.hazardShort(), base.hazardMid(), base.hazardLong()};
    double dHazard = BP / lgd;
    for (int v = 0; v < hazard.length; v++) {
      double[] up = hazard.clone();
      double[] down = hazard.clone();
      up[v] += dHazard;
      down[v] -= dHazard;
      double csUp = simulation.cvaOnly(base.withHazards(up[0], up[1], up[2]), paths, seed);
      double csDown = simulation.cvaOnly(base.withHazards(down[0], down[1], down[2]), paths, seed);
      revaluations += 2;
      out.add(keys.counterpartySpreadDelta(v), 0.5 * (csUp - csDown));
    }

    double fx = base.fxSpot();
    double fxUp = simulation.cvaOnly(base.withFxSpot(fx * (1.0 + RELATIVE)), paths, seed);
    double fxDown = simulation.cvaOnly(base.withFxSpot(fx * (1.0 - RELATIVE)), paths, seed);
    revaluations += 2;
    out.add(keys.fxDelta(), 0.5 * (fxUp - fxDown));

    double fxSig = base.fxVol();
    double fxVegaUp = simulation.cvaOnly(base.withFxVol(fxSig * (1.0 + RELATIVE)), paths, seed);
    double fxVegaDown = simulation.cvaOnly(base.withFxVol(fxSig * (1.0 - RELATIVE)), paths, seed);
    revaluations += 2;
    // (up - down) / (2 * rel) already equals dCVA/dsigma * sigma, the FRTB vega convention
    out.add(keys.fxVega(), (fxVegaUp - fxVegaDown) / (2.0 * RELATIVE));

    return new BumpResult(out.build(), revaluations, (System.nanoTime() - start) / 1.0e9);
  }
}
