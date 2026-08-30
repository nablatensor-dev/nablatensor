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

import com.nablatensor.engine.SDouble;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.GbmPath;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Product;
import com.nablatensor.quant.TimeGrid;
import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.Products;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Seam 1 — the payoff is a lambda. Swapping it re-records the tape in
 * microseconds and the kernel regenerates; the engine, the driver and the Greek
 * machinery never change.
 *
 * <p>The last entry is a bespoke payoff written inline — a capped call,
 * {@code min(max(S_T - K, 0), cap)} — to show that "not in the catalogue" costs
 * three lines, not a fork.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.SwapThePayoff}
 */
public final class SwapThePayoff {

  private SwapThePayoff() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    int steps = Integer.getInteger("steps", 128);
    long scenarios = Long.getLong("scenarios", 1_000_000L);
    long seed = Long.getLong("seed", 42L);

    Map<String, Product<EquityMarket>> book = new LinkedHashMap<>();
    book.put("European call", Products.europeanCall());
    book.put("Asian call", Products.asianCall());
    book.put("Lookback call", Products.lookbackCall());
    book.put("Capped call (inline)", cappedCall(market.spot() * 0.15));

    System.out.printf(Locale.ROOT, "%d steps · %,d scenarios · seed %d · engine cpu-jit%n%n", steps, scenarios, seed);
    System.out.printf(Locale.ROOT, "%-24s %8s %14s %12s %12s%n", "payoff", "nodes", "price", "delta", "vega");

    for (Map.Entry<String, Product<EquityMarket>> entry : book.entrySet()) {
      try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(entry.getValue())
          .market(market).steps(steps).fp64().greeks().on("cpu-jit").build()) {
        Nabla.TypedValuation<EquityMarket> p = mc.run(scenarios, seed);
        System.out.printf(Locale.ROOT, "%-24s %8d %14.6f %12.6f %12.4f%n",
            entry.getKey(), mc.nodes(), p.price(), p.greek(EquityMarket::spot), p.greek(EquityMarket::vol));
      }
    }
  }

  /** A capped call as a raw payoff lambda: {@code min(max(S_T - K, 0), cap)}, discounted. */
  static Product<EquityMarket> cappedCall(double cap) {
    return (rec, in, grid) -> {
      SDouble spot = in.of(EquityMarket::spot);
      SDouble strike = in.of(EquityMarket::strike);
      SDouble rate = in.of(EquityMarket::rate);
      SDouble vol = in.of(EquityMarket::vol);
      SDouble maturity = in.of(EquityMarket::maturity);

      GbmPath model = new GbmPath(rec, rate, vol, grid, maturity);
      SDouble s = spot;
      for (int t = 0; t < grid.steps(); t++) {
        s = model.step(s, rec.randn(), t);
      }
      SDouble payoff = s.sub(strike).max(0.0).min(rec.constant(cap));
      rec.output(payoff.mul(rate.neg().mul(maturity).exp()));
    };
  }
}
