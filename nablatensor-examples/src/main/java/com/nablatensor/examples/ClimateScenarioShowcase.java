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
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Products;
import com.nablatensor.scenario.Scenario;
import com.nablatensor.scenario.ScenarioRunner;
import com.nablatensor.scenario.ScenarioSet;
import com.nablatensor.scenario.Shock;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * {@link StressTestShowcase}'s harness, unchanged, fed a longer horizon and a
 * different macro-scenario set: three NGFS-style transition pathways (orderly,
 * disorderly, hot-house-world) instead of adverse/severely-adverse, and
 * five-year checkpoints out to 2050 instead of quarters. No new calculator:
 * the same record-once / replay-many grid with different scenario data.
 *
 * <p>Unlike the stress-test page, remaining maturity is held fixed at one year
 * at every checkpoint: this book is a steadily-rolled short-dated position
 * being re-marked under each pathway's cumulative drift, not one trade running
 * down to expiry over 25 years.
 *
 * <p>The three pathway curves below are illustrative shapes chosen to show the
 * qualitative NGFS story (orderly = smooth and early, disorderly = abrupt and
 * back-loaded, hot-house-world = continuous physical drag) — they are not the
 * published NGFS macro-variable paths.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.ClimateScenarioShowcase}
 */
public final class ClimateScenarioShowcase {

  private static final double CALL_NOTIONAL = 5_000_000.0;
  private static final double PUT_NOTIONAL = 3_000_000.0;
  private static final int HORIZON_YEARS = 25;
  private static final int[] CHECKPOINTS = {0, 5, 10, 15, 20, 25};

  private ClimateScenarioShowcase() {
  }

  private record Pathway(String name, double spotTarget, double volTarget, double rateTarget, double power) {
    Scenario at(int years) {
      double f = Math.pow((double) years / HORIZON_YEARS, power);
      return Scenario.of(name + "@" + (2025 + years),
          Shock.relative("spot", spotTarget * f),
          Shock.additive("vol", volTarget * f),
          Shock.additive("rate", rateTarget * f));
    }
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    long scenarios = Long.getLong("scenarios", 500_000L);
    long seed = Long.getLong("seed", 42L);
    long bankPositions = Long.getLong("bankPositions", 5_000L);
    String engine = System.getProperty("engine", "cpu-jit");

    double callUnits = CALL_NOTIONAL / market.spot();
    double putUnits = PUT_NOTIONAL / market.spot();

    List<Pathway> pathways = List.of(
        new Pathway("orderly", -0.15, 0.03, 0.010, 1.0),
        new Pathway("disorderly", -0.30, 0.15, 0.020, 2.0),
        new Pathway("hot-house-world", -0.35, 0.10, -0.015, 1.5));

    List<Scenario> scenarioList = new ArrayList<>();
    for (Pathway p : pathways) {
      for (int years : CHECKPOINTS) {
        scenarioList.add(p.at(years));
      }
    }
    ScenarioSet grid = new ScenarioSet(scenarioList);

    System.out.printf(Locale.ROOT, "Climate scenario showcase: same book as the stress-test page, %d NGFS-style "
        + "pathways x %d checkpoints out to %d = %d full-revaluation nodes%n",
        pathways.size(), CHECKPOINTS.length, 2025 + HORIZON_YEARS, grid.size());
    System.out.printf(Locale.ROOT, "%,d scenarios/node, engine %s, seed %d%n%n", scenarios, engine, seed);

    try (MonteCarlo<EquityMarket> callPricer = MonteCarlo.of(Products.europeanCall())
           .market(market).steps(1).fp64().on(engine).build();
         MonteCarlo<EquityMarket> putPricer = MonteCarlo.of(Products.europeanPut())
           .market(market).steps(1).fp64().on(engine).build();
         MonteCarlo<EquityMarket> callGreeks = MonteCarlo.of(Products.europeanCall())
           .market(market).steps(1).fp64().greeks().on(engine).build()) {
      long warm = Math.min(scenarios, 20_000L);
      callPricer.run(warm, seed);
      putPricer.run(warm, seed);
      callGreeks.run(warm, seed);

      Map<String, Nabla.TypedValuation<EquityMarket>> callPv =
          ScenarioRunner.run(callPricer, market, grid, scenarios, seed);
      Map<String, Nabla.TypedValuation<EquityMarket>> putPv =
          ScenarioRunner.run(putPricer, market, grid, scenarios, seed);
      Map<String, Nabla.TypedValuation<EquityMarket>> callDelta =
          ScenarioRunner.run(callGreeks, market, grid, scenarios, seed);

      double pv0 = portfolioPv(callUnits, callPv.get(pathways.get(0).name() + "@2025"),
          putUnits, putPv.get(pathways.get(0).name() + "@2025"));

      System.out.printf(Locale.ROOT, "%-18s", "loss vs. 2025 mark");
      for (int years : CHECKPOINTS) {
        System.out.printf(Locale.ROOT, "%10d", 2025 + years);
      }
      System.out.println();

      String worstName = null;
      double worstLoss = Double.POSITIVE_INFINITY;
      for (Pathway p : pathways) {
        System.out.printf(Locale.ROOT, "%-18s", p.name());
        for (int years : CHECKPOINTS) {
          String name = p.name() + "@" + (2025 + years);
          double pv = portfolioPv(callUnits, callPv.get(name), putUnits, putPv.get(name));
          double loss = pv - pv0;
          System.out.printf(Locale.ROOT, "%10.0f", loss);
          if (loss < worstLoss) {
            worstLoss = loss;
            worstName = name;
          }
        }
        System.out.println();
      }
      System.out.println();

      System.out.printf(Locale.ROOT, "worst projected loss   %12.0f  at %s%n", worstLoss, worstName);
      System.out.printf(Locale.ROOT, "call-leg spot delta there, from the same replay: %.2f%n%n",
          callDelta.get(worstName).greek(EquityMarket::spot));

      double perPositionSeconds = (totalSeconds(callPv) + totalSeconds(putPv)) / 2.0;
      double projectedSeconds = perPositionSeconds * bankPositions;
      System.out.printf(Locale.ROOT, "one position's %d-node grid  %s%n", grid.size(), wall(perPositionSeconds));
      System.out.printf(Locale.ROOT, "%,d-position book, same grid %s%n%n", bankPositions, wall(projectedSeconds));

      System.out.println("Same ScenarioRunner replay as the stress-test page, no new calculator: only the");
      System.out.println("macro-scenario set (NGFS pathways instead of adverse/severely-adverse) and the");
      System.out.println("horizon (decades of checkpoints instead of quarters) changed. The real regulatory");
      System.out.println("grid steps annually, not every five years — this is a scaled-down illustration of");
      System.out.println("the same grid shape, and the pathway curves above are illustrative, not the");
      System.out.println("published NGFS macro-variable paths.");
      System.out.println("What this does not cover: climate-economic scenario construction, sector");
      System.out.println("transition modelling, physical-hazard mapping and counterparty-level emissions");
      System.out.println("data — those sit outside a pricing engine.");
    }
  }

  private static double portfolioPv(double callUnits, Nabla.TypedValuation<EquityMarket> call,
                                    double putUnits, Nabla.TypedValuation<EquityMarket> put) {
    return callUnits * call.price() - putUnits * put.price();
  }

  private static double totalSeconds(Map<String, Nabla.TypedValuation<EquityMarket>> results) {
    return results.values().stream().mapToDouble(Nabla.TypedValuation::seconds).sum();
  }

  private static String wall(double seconds) {
    if (seconds < 90.0) {
      return String.format(Locale.ROOT, "%.3f s", seconds);
    }
    if (seconds < 5_400.0) {
      return String.format(Locale.ROOT, "%.1f min", seconds / 60.0);
    }
    return String.format(Locale.ROOT, "%.2f h", seconds / 3_600.0);
  }
}
