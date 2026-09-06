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

import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.OptionType;
import com.nablatensor.quant.SchwartzMarket;
import com.nablatensor.quant.SchwartzOneFactor;
import com.nablatensor.quant.Seasonality;
import com.nablatensor.quant.SpreadMarket;
import com.nablatensor.quant.SpreadProducts;
import com.nablatensor.quant.analytic.KirkSpreadOption;
import com.nablatensor.quant.analytic.Margrabe;
import java.util.Locale;

/**
 * Feature F10 — commodity and spread pricing. A spark spread (power vs gas) is
 * priced by Kirk's approximation and by a correlated-GBM Monte-Carlo that also
 * returns both leg deltas from one adjoint sweep; then a Schwartz one-factor
 * futures curve is printed with a seasonality overlay.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.SparkSpreadShowcase}
 */
public final class SparkSpreadShowcase {

  private SparkSpreadShowcase() {
  }

  public static void main(String[] args) {
    long paths = Long.getLong("paths", 1_000_000L);

    SpreadMarket m = new SpreadMarket(60.0, 45.0, 0.35, 0.30, 0.0, 0.0, 0.03);
    double rho = 0.55;
    double k = 6.0;
    double t = 0.5;

    double kirk = KirkSpreadOption.price(m.s1(), m.s2(), k, m.vol1(), m.vol2(), rho, m.rate(),
        m.yield1(), m.yield2(), t);
    double margrabe = Margrabe.of(m.s1(), m.s2(), m.vol1(), m.vol2(), rho, 0.0, 0.0, t).price();

    var greeks = adjoint(m, SpreadProducts.spreadOption(k, rho, t, 64), paths);
    System.out.printf(Locale.ROOT, "Spark spread option  (power %.0f, gas-eq %.0f, K=%.0f, %.0fm)%n%n",
        m.s1(), m.s2(), k, t * 12);
    System.out.printf(Locale.ROOT, "  Kirk approximation   %8.4f%n", kirk);
    System.out.printf(Locale.ROOT, "  Monte-Carlo          %8.4f%n", greeks[0]);
    System.out.printf(Locale.ROOT, "  (Margrabe, K=0)      %8.4f%n", margrabe);
    System.out.printf(Locale.ROOT, "  one adjoint sweep:   dS1 = %+.4f   dS2 = %+.4f%n%n", greeks[1], greeks[2]);

    // Schwartz one-factor futures curve with a seasonal overlay.
    SchwartzMarket sch = new SchwartzMarket(50.0, 1.4, Math.log(56.0), 0.28, 0.03);
    Seasonality season = new Seasonality(new double[] {0.06, 0.0}, new double[] {0.03, 0.0});
    System.out.printf(Locale.ROOT, "Schwartz one-factor futures curve (kappa=%.1f, long-run %.1f):%n", sch.kappa(),
        Math.exp(sch.level()));
    System.out.printf(Locale.ROOT, "  %-8s %12s %14s%n", "T (yr)", "futures", "+ seasonality");
    for (double tt : new double[] {0.25, 0.5, 1.0, 2.0, 5.0}) {
      double f = SchwartzOneFactor.futuresPrice(sch, tt);
      System.out.printf(Locale.ROOT, "  %-8.2f %12.4f %14.4f%n", tt, f, f * Math.exp(season.value(tt)));
    }
  }

  private static double[] adjoint(SpreadMarket m,
      java.util.function.BiConsumer<com.nablatensor.engine.AadRecorder, Nabla.Inputs<SpreadMarket>> v,
      long paths) {
    try (Nabla.TypedPricer<SpreadMarket> p = Nabla.model(m, v).fp64().greeks().on("cpu-jit").build()) {
      Nabla.TypedValuation<SpreadMarket> val = p.value().with(m).scenarios(paths).seed(42L).run();
      SpreadMarket g = val.greeks();
      return new double[] {val.price(), g.s1(), g.s2()};
    }
  }
}
