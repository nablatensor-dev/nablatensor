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

import com.nablatensor.quant.CurveSet;
import com.nablatensor.quant.MultiCurveBootstrap;
import java.util.Locale;

/**
 * Feature F5 — a post-LIBOR curve stack. An OIS discount curve and a 3M forecast
 * curve are bootstrapped together; the recorded recursion also yields the exact
 * {@code d(zero rate) / d(quote)} Jacobian from one adjoint sweep. The output
 * shows the tenor basis between the curves and the block lower-triangular
 * Jacobian — a forecast zero rate reacts to the OIS quotes that move its
 * discounting, but an OIS zero never reacts to a forecast quote.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.MultiCurveBootstrapShowcase}
 */
public final class MultiCurveBootstrapShowcase {

  private MultiCurveBootstrapShowcase() {
  }

  public static void main(String[] args) {
    double[] oisPar = {0.0300, 0.0315, 0.0325, 0.0332, 0.0338};
    double[] fwd3mPar = {0.0326, 0.0342, 0.0353, 0.0361, 0.0368};   // ~25bp 3M-OIS basis

    MultiCurveBootstrap.Builder b = MultiCurveBootstrap.builder();
    for (int i = 0; i < oisPar.length; i++) {
      b.oisSwap(i + 1, oisPar[i]);
      b.forecastSwap("3M", i + 1, fwd3mPar[i]);
    }
    MultiCurveBootstrap.Result r = b.build().solve();
    CurveSet cs = r.curves();

    System.out.printf(Locale.ROOT, "Bootstrapped OIS + 3M curves%n%n");
    System.out.printf(Locale.ROOT, "  %-4s %12s %12s %14s%n", "yr", "OIS zero", "3M zero", "basis (bp)");
    for (int n = 1; n <= 5; n++) {
      double zo = cs.discount().zeroRate(n);
      double zf = cs.forecast("3M").zeroRate(n);
      System.out.printf(Locale.ROOT, "  %-4d %11.4f%% %11.4f%% %13.1f%n",
          n, 100 * zo, 100 * zf, 1e4 * (zf - zo));
    }

    // Reprice a 4y receive-fixed swap struck at 3.30% on the stack.
    double strike = 0.0330;
    double annuity = cs.annuity(4);
    double parRate = cs.parSwapRate("3M", 4);
    double pv = (strike - parRate) * annuity;
    System.out.printf(Locale.ROOT, "%n4y receive-fixed swap @ %.2f%%:  par %.4f%%  annuity %.4f  PV %+.5f (per unit notional)%n",
        100 * strike, 100 * parRate, annuity, pv);

    // The 5y 3M-forecast zero's Jacobian row: OIS block then forecast block.
    System.out.printf(Locale.ROOT, "%nJacobian row  d(3M 5y zero) / d(quote):%n");
    String row = "z:3M:swap:5";
    for (String q : r.quoteLabels()) {
      double s = r.sensitivity(row, q);
      if (Math.abs(s) > 1e-9) {
        System.out.printf(Locale.ROOT, "  %-16s %+10.4f%n", q, s);
      }
    }
    System.out.printf(Locale.ROOT, "  (OIS-quote entries are the cross-block terms: discounting risk on a forecast pillar)%n");
  }
}
