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

import com.nablatensor.engine.AadEngines;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Products;
import java.util.Arrays;
import java.util.Locale;

/**
 * Common-random-numbers draw cache vs. plain replay on one CPU engine, same
 * tape, same (seed, pathOffset, paths) repeated every iteration — exactly the
 * shocked-market revaluation pattern {@code -Dnablatensor.crn=on} targets.
 * The first replay of a warmup/measured loop always fills the cache; every
 * later replay of the same block either reuses it (cache on) or regenerates
 * the draws from scratch (cache off).
 *
 * <p>Run: {@code mvn -q -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.bench.CrnBench
 * -Dengine=cpu-jit -Dsteps=252 -Dscenarios=500000
 * -Dnablatensor.crn=on -Dmode=cached}
 */
public final class CrnBench {

  private CrnBench() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    int steps = Integer.getInteger("steps", 252);
    long scenarios = Long.getLong("scenarios", 500_000L);
    long seed = Long.getLong("seed", 42L);
    int warmup = Integer.getInteger("warmup", 3);
    int repeat = Integer.getInteger("repeat", 5);
    String engine = System.getProperty("engine", "cpu-jit");
    String mode = System.getProperty("mode", "unspecified");
    if (steps <= 0 || scenarios <= 0 || warmup < 0 || repeat <= 0) {
      throw new IllegalArgumentException("steps, scenarios and repeat must be positive; warmup must be nonnegative");
    }

    System.out.printf(Locale.ROOT, "# NablaTensor CRN draw-cache bench%n%n");
    System.out.printf(Locale.ROOT, "- machine   : JDK %s, %s %s%n",
        Runtime.version(), System.getProperty("os.name"), System.getProperty("os.arch"));
    System.out.printf(Locale.ROOT, "- engine    : %s   mode: %s   crn: %s%n",
        engine, mode, System.getProperty("nablatensor.crn", "off"));
    System.out.printf(Locale.ROOT, "- product   : Asian call, %d fixings, fp64%n", steps);
    System.out.printf(Locale.ROOT, "- scenarios : %,d   seed : %d   warmup: %d   repeat: %d%n%n",
        scenarios, seed, warmup, repeat);

    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.asianCall())
        .market(market).steps(steps).fp64().greeks().on(engine).build()) {
      System.out.printf(Locale.ROOT, "device: %s%n",
          AadEngines.require(mc.engine(), AadOptions.defaults()).describe());
      System.out.printf(Locale.ROOT, "nodes: %d   build_s: %.3f%n", mc.nodes(), mc.buildSeconds());

      for (int i = 0; i < warmup; i++) {
        mc.run(scenarios, seed);
      }
      Nabla.TypedValuation<EquityMarket> baseline = mc.run(scenarios, seed);

      double bestSeconds = Double.MAX_VALUE;
      double[] samples = new double[repeat];
      boolean bitExact = true;
      for (int i = 0; i < repeat; i++) {
        Nabla.TypedValuation<EquityMarket> r = mc.run(scenarios, seed);
        samples[i] = r.seconds();
        bestSeconds = Math.min(bestSeconds, r.seconds());
        if (Double.compare(r.price(), baseline.price()) != 0) {
          bitExact = false;
        }
      }
      Arrays.sort(samples);
      double medianSeconds = (samples[(repeat - 1) / 2] + samples[repeat / 2]) / 2.0;

      System.out.printf(Locale.ROOT,
          "RESULT engine=%s mode=%s nodes=%d build_s=%.3f settled_s=%.6f median_s=%.6f scen_per_s=%.3e "
              + "price=%.17g delta=%.17g repeat_price_exact=%b%n",
          engine, mode, mc.nodes(), mc.buildSeconds(), bestSeconds, medianSeconds, scenarios / bestSeconds,
          baseline.price(), baseline.greek(EquityMarket::spot), bitExact);
    }
  }
}
