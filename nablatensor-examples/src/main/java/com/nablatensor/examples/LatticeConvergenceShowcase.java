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

import com.nablatensor.lattice.BinomialTree;
import com.nablatensor.lattice.LatticeGreeks;
import com.nablatensor.lattice.LatticePayoff.ExerciseSchedule;
import com.nablatensor.quant.OptionTypeEnum;
import com.nablatensor.quant.analytic.GeneralizedBsm;
import java.util.Locale;

/**
 * Feature F11 — the lattice companion. A binomial tree converging to
 * Black-Scholes (CRR oscillating, Leisen-Reimer smooth), an early-exercise
 * premium the Monte-Carlo engine cannot compute, and tree Greeks.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.LatticeConvergenceShowcase}
 */
public final class LatticeConvergenceShowcase {

  private LatticeConvergenceShowcase() {
  }

  public static void main(String[] args) {
    double s = 100, k = 100, r = 0.05, q = 0.0, vol = 0.20, t = 1.0;
    double closed = GeneralizedBsm.of().type(OptionTypeEnum.CALL).spot(s).strike(k).maturity(t).rate(r).dividend(q).vol(vol).build().price();

    System.out.printf(Locale.ROOT, "European call, Black-Scholes = %.6f%n%n", closed);
    System.out.printf(Locale.ROOT, "  %-6s %14s %14s%n", "steps", "CRR error", "Leisen-Reimer error");
    for (int n : new int[] {10, 25, 50, 100, 250, 500}) {
      double crr = BinomialTree.of().spot(s).rate(r).dividendYield(q).vol(vol).maturity(t).steps(n).method(BinomialTree.MethodEnum.CRR).build()
          .priceVanilla(OptionTypeEnum.CALL, k, ExerciseSchedule.EUROPEAN);
      double lr = BinomialTree.of().spot(s).rate(r).dividendYield(q).vol(vol).maturity(t).steps(n).method(BinomialTree.MethodEnum.LEISEN_REIMER).build()
          .priceVanilla(OptionTypeEnum.CALL, k, ExerciseSchedule.EUROPEAN);
      System.out.printf(Locale.ROOT, "  %-6d %+14.2e %+14.2e%n", n, crr - closed, lr - closed);
    }

    // Early exercise: an American put, which Monte-Carlo can only lower-bound.
    double eur = BinomialTree.of().spot(40).rate(0.06).dividendYield(0.0).vol(0.20).maturity(1.0).steps(2000).method(BinomialTree.MethodEnum.CRR).build()
        .priceVanilla(OptionTypeEnum.PUT, 40, ExerciseSchedule.EUROPEAN);
    double amer = BinomialTree.of().spot(40).rate(0.06).dividendYield(0.0).vol(0.20).maturity(1.0).steps(2000).method(BinomialTree.MethodEnum.CRR).build()
        .priceVanilla(OptionTypeEnum.PUT, 40, ExerciseSchedule.AMERICAN);
    System.out.printf(Locale.ROOT, "%nAmerican put (S=K=40, sigma=0.20, T=1, r=0.06):%n");
    System.out.printf(Locale.ROOT, "  European %.4f   American %.4f   early-exercise premium %.4f%n",
        eur, amer, amer - eur);

    LatticeGreeks g = LatticeGreeks.vanilla(s, r, q, vol, t, 800, BinomialTree.MethodEnum.CRR,
        OptionTypeEnum.CALL, k, ExerciseSchedule.EUROPEAN);
    var ref = GeneralizedBsm.of().type(OptionTypeEnum.CALL).spot(s).strike(k).maturity(t).rate(r).dividend(q).vol(vol).build().greeks();
    System.out.printf(Locale.ROOT, "%nTree Greeks vs closed form:  delta %.4f/%.4f   gamma %.5f/%.5f   vega %.3f/%.3f%n",
        g.delta(), ref.delta(), g.gamma(), ref.gamma(), g.vega(), ref.vega());
  }
}
