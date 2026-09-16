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

import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Products;
import java.util.Locale;

/**
 * Single-backend throughput probe for the JDK-version comparison. Unlike
 * {@link Benchmarks}, this never enumerates {@code AadEngines.available()} —
 * it pins exactly the engine named by {@code -Dengine=...} (cpu-jit or simd),
 * so it never constructs or probes the GPU backends.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.bench.JdkSpeedBench -Dengine=cpu-jit -Dscenarios=2000000}
 */
public final class JdkSpeedBench {

  private JdkSpeedBench() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    int steps = Integer.getInteger("steps", 252);
    long scenarios = Long.getLong("scenarios", 2_000_000L);
    long seed = Long.getLong("seed", 42L);
    String engine = System.getProperty("engine", "cpu-jit");

    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.asianCall())
        .market(market).steps(steps).fp64().greeks().on(engine).build()) {
      mc.run(scenarios, seed);   // warm-up
      Nabla.TypedValuation<EquityMarket> p = mc.run(scenarios, seed);
      System.out.printf(Locale.ROOT,
          "RESULT jdk=%s engine=%s scenarios=%d steps=%d seconds=%.6f scenariosPerSec=%.6e price=%.6f%n",
          Runtime.version(), engine, scenarios, steps, p.seconds(), p.scenariosPerSecond(), p.price());
    }
  }
}
