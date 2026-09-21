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

import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.ExoticProducts;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.OptionTypeEnum;
import com.nablatensor.quant.Product;
import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.Products;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * {@link ProductBench} at {@code .fp32()} for a single named engine, so
 * {@code cpu-jit}, {@code simd} and {@code vulkan} can be compared on the same
 * precision and scenario count. The engine is always pinned with
 * {@code .on(engine)} — this never calls {@code AadEngines.available()}, so it
 * cannot probe a backend that wasn't asked for (vulkan shares the GPU that
 * drives the desktop display; the default scenario count is kept modest for
 * that reason).
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-bench exec:java
 * -Dexec.mainClass=com.nablatensor.bench.Fp32ProductBench
 * -Dengine=cpu-jit -Dscenarios=500000 -Dsteps=128}
 */
public final class Fp32ProductBench {

  private static final int GREEKS = 5;

  private Fp32ProductBench() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    int steps = Integer.getInteger("steps", 128);
    long scenarios = Long.getLong("scenarios", 500_000L);
    long seed = Long.getLong("seed", 42L);
    String engine = System.getProperty("engine", "cpu-jit");

    Map<String, Product<EquityMarket>> book = new LinkedHashMap<>();
    book.put("European call", Products.europeanCall());
    book.put("Asian call", Products.asianCall());
    book.put("Lookback call", Products.lookbackCall());
    book.put("Floating lookback", Products.floatingLookbackCall());
    book.put("BarrierEnum UO call", ExoticProducts.BarrierOption.of().type(OptionTypeEnum.CALL).kind(ExoticProducts.BarrierEnum.UP_OUT).barrier(130).width(1.0).build());
    book.put("Digital cash", ExoticProducts.DigitalCash.of().type(OptionTypeEnum.CALL).cash(1.0).width(1.0).build());
    book.put("Cliquet", ExoticProducts.Cliquet.of().localFloor(-0.05).localCap(0.05).globalFloor(0.0).globalCap(0.4).notional(100.0).build());
    book.put("Autocallable", ExoticProducts.Autocallable.of().autocallLevel(105.0).couponPerPeriod(0.02).observations(4).width(1.0).notional(100.0).build());

    System.out.printf(Locale.ROOT, "# Per-product: adjoint vs bump  (engine %s, fp32, %,d scenarios, %d steps, seed %d)%n%n",
        engine, scenarios, steps, seed);
    System.out.printf(Locale.ROOT, "| product | price | delta | adjoint | bump (1+2x%d) | speedup |%n", GREEKS);
    System.out.printf(Locale.ROOT, "|---|--:|--:|--:|--:|--:|%n");

    for (Map.Entry<String, Product<EquityMarket>> e : book.entrySet()) {
      try (MonteCarlo<EquityMarket> greeks = MonteCarlo.of(e.getValue()).market(market).steps(steps)
               .fp32().greeks().on(engine).build();
           MonteCarlo<EquityMarket> price = MonteCarlo.of(e.getValue()).market(market).steps(steps)
               .fp32().priceOnly().on(engine).build()) {

        greeks.run(scenarios, seed);
        greeks.run(scenarios, seed);
        double adjSec = best(3, () -> greeks.run(scenarios, seed).seconds());
        Nabla.TypedValuation<EquityMarket> g = greeks.run(scenarios, seed);

        price.run(scenarios, seed);
        price.run(scenarios, seed);
        double oneReval = best(3, () -> price.run(scenarios, seed).seconds());
        double bumpSec = (1 + 2 * GREEKS) * oneReval;

        System.out.printf(Locale.ROOT, "| %s | %.5f | %.5f | %.4f s | %.4f s | %.1fx |%n",
            e.getKey(), g.price(), g.greek(EquityMarket::spot), adjSec, bumpSec, bumpSec / adjSec);
      }
    }
  }

  private static double best(int rounds, java.util.function.DoubleSupplier body) {
    double b = Double.MAX_VALUE;
    for (int i = 0; i < rounds; i++) {
      b = Math.min(b, body.getAsDouble());
    }
    return b;
  }
}
