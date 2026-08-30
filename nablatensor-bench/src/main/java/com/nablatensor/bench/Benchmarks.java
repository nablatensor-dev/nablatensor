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
package com.nablatensor.bench;

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadEngines;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Products;
import com.nablatensor.engine.Nabla;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Reproducible comparison harness. Prints the machine, the seed and two tables:
 *
 * <ol>
 *   <li><b>adjoint vs bump-and-revalue</b> — one adjoint sweep returns value plus
 *       every Greek; the classic alternative is {@code 1 + 2N} price-only
 *       revaluations for an {@code N}-Greek central difference. Same numbers,
 *       measured cost ratio.</li>
 *   <li><b>backend matrix</b> — one recorded tape replayed on every backend this
 *       machine can run, scenarios per second.</li>
 * </ol>
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-bench exec:java
 * -Dexec.mainClass=com.nablatensor.bench.Benchmarks -Dscenarios=2000000}
 */
public final class Benchmarks {

  private static final int GREEKS = 5;   // delta, dV/dK, vega, rho, dV/dT

  private Benchmarks() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    int steps = Integer.getInteger("steps", 252);
    long scenarios = Long.getLong("scenarios", 1_000_000L);
    long seed = Long.getLong("seed", 42L);
    String engine = System.getProperty("engine", "cpu-jit");

    System.out.printf(Locale.ROOT, "# NablaTensor benchmark%n%n");
    System.out.printf(Locale.ROOT, "- machine   : JDK %s, %s %s, %d processors%n",
        Runtime.version(), System.getProperty("os.name"), System.getProperty("os.arch"),
        Runtime.getRuntime().availableProcessors());
    System.out.printf(Locale.ROOT, "- product   : Asian call, %d fixings, fp64%n", steps);
    System.out.printf(Locale.ROOT, "- scenarios : %,d   seed : %d%n%n", scenarios, seed);

    adjointVsBump(market, steps, scenarios, seed, engine);
    backendMatrix(market, steps, scenarios, seed);
  }

  private static void adjointVsBump(EquityMarket market, int steps, long scenarios, long seed,
                                    String engine) {
    try (MonteCarlo<EquityMarket> adjoint = MonteCarlo.of(Products.asianCall())
             .market(market).steps(steps).fp64().greeks().on(engine).build();
         MonteCarlo<EquityMarket> priceOnly = MonteCarlo.of(Products.asianCall())
             .market(market).steps(steps).fp64().priceOnly().on(engine).build()) {

      adjoint.run(scenarios, seed);
      double adjointSec = bestOf(3, () -> adjoint.run(scenarios, seed).seconds());
      Nabla.TypedValuation<EquityMarket> g = adjoint.run(scenarios, seed);

      priceOnly.run(scenarios, seed);
      double oneRevalSec = bestOf(3, () -> priceOnly.run(scenarios, seed).seconds());
      double bumpSec = (1 + 2 * GREEKS) * oneRevalSec;

      System.out.printf(Locale.ROOT, "## adjoint vs bump-and-revalue  (engine %s)%n%n", engine);
      System.out.printf(Locale.ROOT, "| method | replays | wall clock | speedup |%n");
      System.out.printf(Locale.ROOT, "|---|--:|--:|--:|%n");
      System.out.printf(Locale.ROOT, "| adjoint (value + %d Greeks, one sweep) | 1 | %.4f s | %.1fx |%n",
          GREEKS, adjointSec, bumpSec / adjointSec);
      System.out.printf(Locale.ROOT, "| central bump (1 + 2x%d price-only) | %d | %.4f s | 1.0x |%n%n",
          GREEKS, 1 + 2 * GREEKS, bumpSec);
      System.out.printf(Locale.ROOT, "adjoint: price=%.6f delta=%.6f vega=%.6f rho=%.6f%n%n",
          g.price(), g.greek(EquityMarket::spot), g.greek(EquityMarket::vol), g.greek(EquityMarket::rate));
    }
  }

  private static void backendMatrix(EquityMarket market, int steps, long scenarios, long seed) {
    System.out.printf(Locale.ROOT, "## backend matrix  (one tape, same seed)%n%n");
    System.out.printf(Locale.ROOT, "| engine | price | delta | scenarios/s | runs on |%n");
    System.out.printf(Locale.ROOT, "|---|--:|--:|--:|---|%n");
    for (AadEngine e : available()) {
      String name = e.name();
      try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.asianCall())
          .market(market).steps(steps).fp64().greeks().on(name).build()) {
        mc.run(scenarios, seed);
        Nabla.TypedValuation<EquityMarket> p = bestPricing(3, mc, scenarios, seed);
        System.out.printf(Locale.ROOT, "| %s | %.6f | %.6f | %.2e | %s |%n",
            name, p.price(), p.greek(EquityMarket::spot), p.scenariosPerSecond(), e.describe());
      } catch (RuntimeException | LinkageError ex) {
        System.out.printf(Locale.ROOT, "| %s | — | — | — | skipped: %s |%n", name, ex.getMessage());
      }
    }
    System.out.println();
  }

  private static List<AadEngine> available() {
    return new ArrayList<>(AadEngines.available(new AadOptions(AadOptions.Precision.FLOAT64, true)));
  }

  private static double bestOf(int rounds, DoubleSupplier body) {
    double best = Double.MAX_VALUE;
    for (int i = 0; i < rounds; i++) {
      best = Math.min(best, body.getAsDouble());
    }
    return best;
  }

  private static Nabla.TypedValuation<EquityMarket> bestPricing(int rounds, MonteCarlo<EquityMarket> mc, long scenarios, long seed) {
    Nabla.TypedValuation<EquityMarket> best = null;
    for (int i = 0; i < rounds; i++) {
      Nabla.TypedValuation<EquityMarket> p = mc.run(scenarios, seed);
      if (best == null || p.seconds() < best.seconds()) {
        best = p;
      }
    }
    return best;
  }

  @FunctionalInterface
  private interface DoubleSupplier {
    double getAsDouble();
  }
}
