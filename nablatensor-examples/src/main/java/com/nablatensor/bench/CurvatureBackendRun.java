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

/** Compares the FRTB curvature showcase workflow on four execution backends. */
public final class CurvatureBackendRun {

  private static final String[] ENGINES = {"cpu", "cpu-jit", "simd", "cuda"};
  private static final double SHORT_POSITION = -1.0;
  private static final double SHOCK_FRACTION = 0.30;

  private CurvatureBackendRun() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    int steps = Integer.getInteger("steps", 252);
    long scenarios = Long.getLong("scenarios", 1_000_000L);
    long seed = Long.getLong("seed", 42L);
    int rounds = Integer.getInteger("rounds", 3);

    System.out.printf(Locale.ROOT, "# FRTB curvature backend comparison%n%n");
    System.out.printf(Locale.ROOT, "Asian call, %,d paths x %d fixings, seed %d, best of %d%n%n",
        scenarios, steps, seed, rounds);
    System.out.println("| backend | precision | adjoint | 3 price replays | replay vs cpu | CVR | workflow | vs cpu |");
    System.out.println("|---|---:|--:|--:|--:|--:|--:|--:|");

    double cpuWorkflow = Double.NaN;
    double cpuRepricing = Double.NaN;
    for (String engine : ENGINES) {
      boolean fp32 = engine.equals("cuda");
      try (MonteCarlo<EquityMarket> greeks = configure(MonteCarlo.of(Products.asianCall())
           .market(market).steps(steps).greeks().on(engine), fp32).build();
         MonteCarlo<EquityMarket> pricer = configure(MonteCarlo.of(Products.asianCall())
           .market(market).steps(steps).priceOnly().on(engine), fp32).build()) {
        long warmScenarios = Math.min(scenarios, 100_000L);
        greeks.run(warmScenarios, seed);
        pricer.run(warmScenarios, seed);

        Nabla.TypedValuation<EquityMarket> risk = best(rounds, greeks, market, scenarios, seed);
        Nabla.TypedValuation<EquityMarket> base = best(rounds, pricer, market, scenarios, seed);
        Nabla.TypedValuation<EquityMarket> up = best(rounds, pricer,
            market.withSpot(market.spot() * (1.0 + SHOCK_FRACTION)), scenarios, seed);
        Nabla.TypedValuation<EquityMarket> down = best(rounds, pricer,
            market.withSpot(market.spot() * (1.0 - SHOCK_FRACTION)), scenarios, seed);

        double repricingSeconds = base.seconds() + up.seconds() + down.seconds();
        double workflowSeconds = risk.seconds() + repricingSeconds;
        if (engine.equals("cpu")) {
          cpuWorkflow = workflowSeconds;
          cpuRepricing = repricingSeconds;
        }
        double delta = SHORT_POSITION * risk.greek(EquityMarket::spot);
        double cvr = curvature(
            SHORT_POSITION * base.price(), SHORT_POSITION * up.price(),
            SHORT_POSITION * down.price(), SHOCK_FRACTION * market.spot(), delta);
        String speedup = Double.isNaN(cpuWorkflow)
            ? "-" : String.format(Locale.ROOT, "%.2fx", cpuWorkflow / workflowSeconds);
        String replaySpeedup = Double.isNaN(cpuRepricing)
          ? "-" : String.format(Locale.ROOT, "%.2fx", cpuRepricing / repricingSeconds);

        System.out.printf(Locale.ROOT, "| %s | %s | %.4f s | %.4f s | %s | %.6f | %.4f s | %s |%n",
            engine, fp32 ? "fp32" : "fp64", risk.seconds(), repricingSeconds,
          replaySpeedup, cvr, workflowSeconds, speedup);
      } catch (RuntimeException | LinkageError error) {
        System.out.printf("| %s | | skipped: %s | | | | | |%n", engine, error.getMessage());
      }
    }
  }

  private static MonteCarlo.Builder<EquityMarket> configure(
      MonteCarlo.Builder<EquityMarket> builder, boolean fp32) {
    return fp32 ? builder.fp32() : builder.fp64();
  }

  private static Nabla.TypedValuation<EquityMarket> best(
      int rounds, MonteCarlo<EquityMarket> monteCarlo, EquityMarket market,
      long scenarios, long seed) {
    Nabla.TypedValuation<EquityMarket> best = null;
    for (int round = 0; round < rounds; round++) {
      Nabla.TypedValuation<EquityMarket> result = monteCarlo.run(market, scenarios, seed);
      if (best == null || result.seconds() < best.seconds()) {
        best = result;
      }
    }
    return best;
  }

  private static double curvature(double base, double up, double down,
                                  double shock, double delta) {
    return -Math.min(up - base - shock * delta, down - base + shock * delta);
  }
}