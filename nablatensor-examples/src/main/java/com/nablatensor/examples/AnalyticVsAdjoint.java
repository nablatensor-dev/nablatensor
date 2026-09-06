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

import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.OptionType;
import com.nablatensor.quant.Products;
import com.nablatensor.quant.analytic.AnalyticGreeks;
import com.nablatensor.quant.analytic.Black76;
import com.nablatensor.quant.analytic.GeneralizedBsm;
import com.nablatensor.engine.Nabla;
import com.nablatensor.validate.ModelValidation;
import com.nablatensor.validate.Report;
import java.util.Locale;

/**
 * Feature F2 — the analytic oracle layer. The same at-the-money European is
 * priced three ways that must agree: the adjoint Monte-Carlo sweep, the
 * generalised Black-Scholes-Merton closed form ({@link GeneralizedBsm}), and
 * Black's 1976 model on the forward ({@link Black76}). Then the
 * {@link ModelValidation} evidence pack is regenerated with the closed form
 * registered as an independent reference.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.AnalyticVsAdjoint}
 */
public final class AnalyticVsAdjoint {

  private AnalyticVsAdjoint() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();      // S0=K=100, sigma=20%, r=3%, T=1y
    long scenarios = Long.getLong("scenarios", 2_000_000L);
    long seed = Long.getLong("seed", 42L);

    AnalyticGreeks bsm = GeneralizedBsm.of(OptionType.CALL,
        market.spot(), market.strike(), market.maturity(), market.rate(), 0.0, market.vol()).greeks();

    double forward = market.spot() * Math.exp(market.rate() * market.maturity());
    AnalyticGreeks black76 = Black76.of(OptionType.CALL,
        forward, market.strike(), market.maturity(), market.rate(), market.vol());

    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.europeanCall())
        .market(market).steps(1).fp64().greeks().on("cpu-jit").build()) {

      Nabla.TypedValuation<EquityMarket> p = mc.run(scenarios, seed);

      System.out.printf(Locale.ROOT, "%,d scenarios  (± %.4f standard error)%n%n", p.scenarios(), p.standardError());
      System.out.printf(Locale.ROOT, "%-8s %14s %14s %14s%n", "", "adjoint MC", "gen. BSM", "Black-76(F)");
      row("price", p.price(), bsm.price(), black76.price());
      row("delta", p.greek(EquityMarket::spot), bsm.delta(), black76.delta());
      row("vega", p.greek(EquityMarket::vol), bsm.vega(), black76.vega());
      row("rho", p.greek(EquityMarket::rate), bsm.rho(), black76.rho());
      row("dV/dK", p.greek(EquityMarket::strike), bsm.strikeSensitivity(), black76.strikeSensitivity());
      System.out.printf(Locale.ROOT,
          "%n(only price and dV/dK are like-for-like across all three: Black-76 delta is dV/dF and%n"
              + " its rho is the pure discounting term -T*price, since the forward F is held fixed.)%n");
    }

    System.out.println();
    Report report = ModelValidation.of(Products.europeanCall())
        .market(market).steps(1)
        .scenarios(Math.min(scenarios, 500_000L)).seed(seed)
        .fp64().tolerance(1e-6)
        .analyticReference(m -> GeneralizedBsm.of(OptionType.CALL,
            m.spot(), m.strike(), m.maturity(), m.rate(), 0.0, m.vol()).greeks())
        .run();
    System.out.println(report);
  }

  private static void row(String name, double mc, double a, double b) {
    System.out.printf(Locale.ROOT, "%-8s %14.6f %14.6f %14.6f%n", name, mc, a, b);
  }
}
