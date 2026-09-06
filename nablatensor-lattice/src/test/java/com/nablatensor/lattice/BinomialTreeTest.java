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
package com.nablatensor.lattice;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.lattice.LatticePayoff.ExerciseSchedule;
import com.nablatensor.quant.OptionType;
import com.nablatensor.quant.analytic.AnalyticGreeks;
import com.nablatensor.quant.analytic.GeneralizedBsm;
import org.junit.jupiter.api.Test;

/**
 * Feature F11: the binomial tree converges to Black-Scholes, prices early
 * exercise, matches its closed-form Greeks, and its Leisen-Reimer variant
 * converges smoothly and fast.
 */
class BinomialTreeTest {

  private static final double S = 100, K = 100, R = 0.05, Q = 0.0, VOL = 0.20, T = 1.0;

  private static double bs(OptionType type, double q) {
    return GeneralizedBsm.of(type, S, K, T, R, q, VOL).price();
  }

  @Test
  void crrEuropeanConvergesToBlackScholes() {
    double closed = bs(OptionType.CALL, Q);
    double coarse = BinomialTree.of(S, R, Q, VOL, T, 25, BinomialTree.Method.CRR)
        .priceVanilla(OptionType.CALL, K, ExerciseSchedule.EUROPEAN);
    double fine = BinomialTree.of(S, R, Q, VOL, T, 2000, BinomialTree.Method.CRR)
        .priceVanilla(OptionType.CALL, K, ExerciseSchedule.EUROPEAN);
    assertTrue(Math.abs(fine - closed) < 5e-3, "CRR@2000 close to BSM, err " + Math.abs(fine - closed));
    assertTrue(Math.abs(fine - closed) < Math.abs(coarse - closed), "finer tree is closer");

    ConvergenceTable ct = ConvergenceTable.crrVanilla(S, R, Q, VOL, T, OptionType.CALL, K,
        ExerciseSchedule.EUROPEAN, new int[] {50, 100, 200, 400, 800});
    assertEquals(closed, ct.richardsonExtrapolated(), 2e-2, "Richardson extrapolation tightens the estimate");
  }

  @Test
  void leisenReimerConvergesSmoothlyAndFast() {
    double closed = bs(OptionType.PUT, Q);
    double prev = Double.NaN;
    for (int n : new int[] {21, 41, 81, 161}) {
      double px = BinomialTree.of(S, R, Q, VOL, T, n, BinomialTree.Method.LEISEN_REIMER)
          .priceVanilla(OptionType.PUT, K, ExerciseSchedule.EUROPEAN);
      if (!Double.isNaN(prev)) {
        assertTrue(Math.abs(px - closed) <= Math.abs(prev - closed) + 1e-9, "monotone (non-oscillating) convergence");
      }
      prev = px;
    }
    assertEquals(closed, prev, 2e-3, "Leisen-Reimer @161 close to BSM");
  }

  @Test
  void europeanPutCallParityHoldsOnTheTree() {
    double q = 0.03;
    BinomialTree tree = BinomialTree.of(S, R, q, VOL, T, 600, BinomialTree.Method.CRR);
    double call = tree.priceVanilla(OptionType.CALL, K, ExerciseSchedule.EUROPEAN);
    double put = tree.priceVanilla(OptionType.PUT, K, ExerciseSchedule.EUROPEAN);
    assertEquals(S * Math.exp(-q * T) - K * Math.exp(-R * T), call - put, 5e-3, "tree put-call parity");
  }

  @Test
  void americanPutMatchesTheHighAccuracyBenchmark() {
    // S=K=40, sigma=0.20, T=1, r=0.06. High-accuracy benchmark ~ 2.3196
    // (Broadie-Detemple); the Longstaff-Schwartz (2001) FD figure of 2.314 is
    // the coarser-grid approximation.
    double px1000 = BinomialTree.of(40, 0.06, 0.0, 0.20, 1.0, 1000, BinomialTree.Method.CRR)
        .priceVanilla(OptionType.PUT, 40, ExerciseSchedule.AMERICAN);
    double px4000 = BinomialTree.of(40, 0.06, 0.0, 0.20, 1.0, 4000, BinomialTree.Method.CRR)
        .priceVanilla(OptionType.PUT, 40, ExerciseSchedule.AMERICAN);
    assertEquals(2.3196, px4000, 3e-3, "American put on a fine CRR tree");
    assertTrue(Math.abs(px4000 - 2.3196) < Math.abs(px1000 - 2.3196) + 1e-3, "converging with steps");
  }

  @Test
  void americanCallEqualsEuropeanWithoutDividendButNotWithOne() {
    BinomialTree noDiv = BinomialTree.of(S, R, 0.0, VOL, T, 800, BinomialTree.Method.CRR);
    assertEquals(
        noDiv.priceVanilla(OptionType.CALL, K, ExerciseSchedule.EUROPEAN),
        noDiv.priceVanilla(OptionType.CALL, K, ExerciseSchedule.AMERICAN),
        1e-9, "no-dividend American call == European call");

    BinomialTree div = BinomialTree.of(S, R, 0.06, VOL, T, 800, BinomialTree.Method.CRR);
    double eur = div.priceVanilla(OptionType.CALL, K, ExerciseSchedule.EUROPEAN);
    double amer = div.priceVanilla(OptionType.CALL, K, ExerciseSchedule.AMERICAN);
    assertTrue(amer > eur + 1e-4, "with a dividend, early exercise of a call has value");
  }

  @Test
  void bermudanSitsBetweenEuropeanAndAmerican() {
    BinomialTree tree = BinomialTree.of(40, 0.06, 0.0, 0.20, 1.0, 600, BinomialTree.Method.CRR);
    double eur = tree.priceVanilla(OptionType.PUT, 40, ExerciseSchedule.EUROPEAN);
    double amer = tree.priceVanilla(OptionType.PUT, 40, ExerciseSchedule.AMERICAN);
    double berm = tree.priceVanilla(OptionType.PUT, 40, ExerciseSchedule.everyNthStep(50));
    assertTrue(eur <= berm + 1e-9 && berm <= amer + 1e-9, "European <= Bermudan <= American");
    assertTrue(berm > eur + 1e-3, "the exercise windows add value");
  }

  @Test
  void treeGreeksMatchTheClosedForm() {
    AnalyticGreeks ref = GeneralizedBsm.of(OptionType.CALL, S, K, T, R, Q, VOL).greeks();
    LatticeGreeks g = LatticeGreeks.vanilla(S, R, Q, VOL, T, 600, BinomialTree.Method.CRR,
        OptionType.CALL, K, ExerciseSchedule.EUROPEAN);
    assertEquals(ref.price(), g.price(), 5e-3, "price");
    assertEquals(ref.delta(), g.delta(), 5e-3, "delta");
    assertEquals(ref.gamma(), g.gamma(), 2e-3, "gamma");
    assertEquals(ref.vega(), g.vega(), 0.1, "vega");
    assertEquals(ref.rho(), g.rho(), 0.1, "rho");
    assertTrue(g.theta() > 0, "dV/dT positive for this call");
  }
}
