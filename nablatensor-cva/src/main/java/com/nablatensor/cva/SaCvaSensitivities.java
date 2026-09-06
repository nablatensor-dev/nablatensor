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
import java.util.function.DoubleUnaryOperator;

/**
 * The SA-CVA sensitivity vector for a netting set, produced two ways:
 *
 * <ul>
 *   <li><b>Route B — {@link #adjoint}</b>: read straight off the one adjoint
 *       sweep in {@link CvaResult#gradient()}. {@code O(1)} regardless of how
 *       many risk factors the netting set touches.</li>
 *   <li><b>Route A — {@link #bumpAndRevalue}</b>: the letter-compliant
 *       prescribed bump — re-simulate the whole netting-set exposure per shocked
 *       risk factor and take a Richardson-extrapolated central difference (steps
 *       {@code h} and {@code 2h}, four re-simulations per factor). {@code
 *       O(#risk factors)}; this is the cost the adjoint sweep removes.</li>
 * </ul>
 *
 * <p>Both return a {@code Sensitivities} keyed by {@link RiskFactor}, fed
 * straight into {@link SaCva}. On common random numbers the two agree to the
 * bump's own {@code O(h^4)} error — the reconciliation the benchmark rests on.
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
    int[] revaluations = {0};
    double lgd = keys.counterparty().lossGivenDefault();
    Sensitivities.Builder out = Sensitivities.builder();

    // Each factor's slope d(CVA)/d(offset) is a Richardson-extrapolated central
    // difference (steps h and 2h, O(h^4) residual) on common random numbers.
    // The extrapolation lifts the finite difference well clear of the fp32
    // round-off floor, so the bump reconciles with the adjoint sweep even when
    // both run single-precision.
    DoubleUnaryOperator ir = a -> {
      revaluations[0]++;
      return simulation.cvaOnly(base.withCurveLevel(base.r0() + a, base.hwLevel() + a), paths, seed);
    };
    out.add(keys.irDelta(), slope(ir, BP) * BP);

    double sig = base.hwSigma();
    DoubleUnaryOperator rateVol = a -> {
      revaluations[0]++;
      return simulation.cvaOnly(base.withRateVol(sig * (1.0 + a)), paths, seed);
    };
    out.add(keys.irVega(), slope(rateVol, RELATIVE)); // slope in relative units == dCVA/dsigma * sigma

    double[] hazard = {base.hazardShort(), base.hazardMid(), base.hazardLong()};
    double dHazard = BP / lgd;
    for (int v = 0; v < hazard.length; v++) {
      int vertex = v;
      DoubleUnaryOperator cs = a -> {
        revaluations[0]++;
        double[] h = hazard.clone();
        h[vertex] += a;
        return simulation.cvaOnly(base.withHazards(h[0], h[1], h[2]), paths, seed);
      };
      out.add(keys.counterpartySpreadDelta(v), slope(cs, dHazard) * dHazard);
    }

    double fx = base.fxSpot();
    DoubleUnaryOperator fxSpot = a -> {
      revaluations[0]++;
      return simulation.cvaOnly(base.withFxSpot(fx * (1.0 + a)), paths, seed);
    };
    out.add(keys.fxDelta(), slope(fxSpot, RELATIVE) * RELATIVE);

    double fxSig = base.fxVol();
    DoubleUnaryOperator fxVol = a -> {
      revaluations[0]++;
      return simulation.cvaOnly(base.withFxVol(fxSig * (1.0 + a)), paths, seed);
    };
    out.add(keys.fxVega(), slope(fxVol, RELATIVE)); // FRTB vega convention, as above

    return new BumpResult(out.build(), revaluations[0], (System.nanoTime() - start) / 1.0e9);
  }

  /**
   * Richardson-extrapolated central-difference estimate of {@code d f / d a} at
   * {@code a = 0}: {@code (4*D(h) - D(2h)) / 3}, where {@code D(s)} is the
   * two-sided difference at step {@code s}. Cancels the {@code O(h^2)} term of a
   * plain central difference, leaving {@code O(h^4)}. Four evaluations of
   * {@code f}.
   */
  private static double slope(DoubleUnaryOperator f, double h) {
    double dHalf = (f.applyAsDouble(h) - f.applyAsDouble(-h)) / (2.0 * h);
    double dFull = (f.applyAsDouble(2.0 * h) - f.applyAsDouble(-2.0 * h)) / (4.0 * h);
    return (4.0 * dHalf - dFull) / 3.0;
  }
}
