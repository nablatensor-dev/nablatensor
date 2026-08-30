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
 * Arithmetic-average Asian call: one recording, replayed on every backend this
 * machine can run. Same tape, same seed, same numbers to Monte-Carlo noise — the
 * only thing that changes between rows is throughput.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.AsianGreeksBackends}
 */
public final class AsianGreeksBackends {

  private AsianGreeksBackends() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    int steps = Integer.getInteger("steps", 252);
    long scenarios = Long.getLong("scenarios", 1_000_000L);
    long seed = Long.getLong("seed", 42L);

    System.out.printf(Locale.ROOT, "Asian call · %d fixings · %,d scenarios · seed %d · fp64%n%n",
        steps, scenarios, seed);
    System.out.printf(Locale.ROOT, "%-10s %14s %12s %12s %14s  %s%n",
        "engine", "price", "delta", "vega", "scen/s", "runs on");

    for (String engine : availableEngines()) {
      try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.asianCall())
          .market(market).steps(steps).fp64().greeks().on(engine).build()) {

        mc.run(scenarios, seed);                     // warm
        Nabla.TypedValuation<EquityMarket> p = mc.run(scenarios, seed);

        System.out.printf(Locale.ROOT, "%-10s %14.6f %12.6f %12.4f %14.2e  %s%n",
            engine, p.price(), p.greek(EquityMarket::spot), p.greek(EquityMarket::vol), p.scenariosPerSecond(), describe(engine));
      } catch (RuntimeException | LinkageError e) {
        System.out.printf(Locale.ROOT, "%-10s  (skipped: %s)%n", engine, e.getMessage());
      }
    }
  }

  private static List<String> availableEngines() {
    List<String> names = new ArrayList<>();
    for (AadEngine e : AadEngines.available(new AadOptions(AadOptions.Precision.FLOAT64, true))) {
      names.add(e.name());
    }
    return names;
  }

  private static String describe(String engine) {
    return AadEngines.available(new AadOptions(AadOptions.Precision.FLOAT64, true)).stream()
        .filter(e -> e.name().equals(engine))
        .map(AadEngine::describe)
        .findFirst()
        .orElse("");
  }
}
