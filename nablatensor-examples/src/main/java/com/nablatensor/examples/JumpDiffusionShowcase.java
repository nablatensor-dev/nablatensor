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
import com.nablatensor.quant.KouJumpModel;
import com.nablatensor.quant.KouMarket;
import com.nablatensor.quant.MertonJumpMarket;
import com.nablatensor.quant.MertonJumpModel;
import com.nablatensor.quant.OptionType;
import com.nablatensor.quant.analytic.GeneralizedBsm;
import com.nablatensor.quant.analytic.MertonJumpDiffusion;
import java.util.Locale;

/**
 * Feature F7 — jump-diffusion step blocks. A Merton jump-diffusion European call
 * is priced by the recorded Monte-Carlo step block and checked against the exact
 * Poisson-series closed form; then the implied-volatility smile the jumps
 * generate is printed for both the Merton and Kou models, next to the flat
 * Black-Scholes line.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.JumpDiffusionShowcase}
 */
public final class JumpDiffusionShowcase {

  private JumpDiffusionShowcase() {
  }

  public static void main(String[] args) {
    long paths = Long.getLong("paths", 1_500_000L);
    int steps = Integer.getInteger("steps", 128);
    double s0 = 100.0;
    double r = 0.03;
    double t = 0.5;
    double sigma = 0.15;

    MertonJumpMarket merton = new MertonJumpMarket(s0, 100, sigma, r, t, 1.2, -0.09, 0.18);

    double mc = price(merton, MertonJumpModel.european(OptionType.CALL, t, steps), paths, 42L);
    double exact = MertonJumpDiffusion.price(OptionType.CALL, s0, 100, t, r, sigma,
        merton.jumpIntensity(), merton.jumpMean(), merton.jumpVol());
    System.out.printf(Locale.ROOT, "Merton ATM call: MC %.4f   exact series %.4f   (%.0f k paths, %d steps)%n%n",
        mc, exact, paths / 1e3, steps);

    KouMarket kou = new KouMarket(s0, 100, sigma, r, t, 1.2, 0.35, 12.0, 7.0);

    System.out.printf(Locale.ROOT, "Implied volatility smile (%.0fm option):%n", t * 12);
    System.out.printf(Locale.ROOT, "  %-8s %10s %10s %10s%n", "strike", "Black", "Merton", "Kou");
    for (double k : new double[] {80, 90, 100, 110, 120}) {
      double bs = GeneralizedBsm.of(OptionType.CALL, s0, k, t, r, 0.0, sigma).price();
      double pm = price(withStrike(merton, k), MertonJumpModel.european(OptionType.CALL, t, steps), paths, 7L);
      double pk = price(withStrike(kou, k), KouJumpModel.european(OptionType.CALL, t, steps), paths, 7L);
      System.out.printf(Locale.ROOT, "  %-8.0f %10.4f %10.4f %10.4f  | ivol  B %.2f%%  M %.2f%%  K %.2f%%%n",
          k, bs, pm, pk,
          100 * impliedVol(bs, s0, k, t, r), 100 * impliedVol(pm, s0, k, t, r), 100 * impliedVol(pk, s0, k, t, r));
    }
  }

  private static MertonJumpMarket withStrike(MertonJumpMarket m, double k) {
    return new MertonJumpMarket(m.spot(), k, m.vol(), m.rate(), m.maturity(),
        m.jumpIntensity(), m.jumpMean(), m.jumpVol());
  }

  private static KouMarket withStrike(KouMarket m, double k) {
    return new KouMarket(m.spot(), k, m.vol(), m.rate(), m.maturity(),
        m.jumpIntensity(), m.probUp(), m.etaUp(), m.etaDown());
  }

  private static <M extends Record> double price(M market,
      java.util.function.BiConsumer<com.nablatensor.engine.AadRecorder, Nabla.Inputs<M>> v,
      long paths, long seed) {
    try (Nabla.TypedPricer<M> p = Nabla.model(market, v).fp64().priceOnly().on("cpu-jit").build()) {
      return p.value().with(market).scenarios(paths).seed(seed).run().price();
    }
  }

  /** Bisection on the Black-Scholes call price for the implied vol. */
  private static double impliedVol(double price, double s, double k, double t, double r) {
    double lo = 1e-4;
    double hi = 3.0;
    for (int i = 0; i < 100; i++) {
      double mid = 0.5 * (lo + hi);
      double pv = GeneralizedBsm.of(OptionType.CALL, s, k, t, r, 0.0, mid).price();
      if (pv > price) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    return 0.5 * (lo + hi);
  }
}
