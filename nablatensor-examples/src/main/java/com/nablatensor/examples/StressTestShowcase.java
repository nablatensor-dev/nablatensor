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
 * Supervisory stress testing's own shape: a small book, full-revalued on a
 * grid of (macro scenario) &times; (forward horizon) nodes from kernels that
 * were each recorded and compiled exactly once. Two-leg book: a long call and
 * a written (short) put on the same underlying, so a market crash hurts both
 * legs at once, the way a real trading book's crash losses compound.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.StressTestShowcase}
 */
public final class StressTestShowcase {

  private static final double CALL_NOTIONAL = 5_000_000.0;
  private static final double PUT_NOTIONAL = 3_000_000.0;

  private StressTestShowcase() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    long scenarios = Long.getLong("scenarios", 500_000L);
    long seed = Long.getLong("seed", 42L);
    long bankPositions = Long.getLong("bankPositions", 5_000L);
    String engine = System.getProperty("engine", "cpu-jit");

    // Unit count is fixed at trade inception (notional / spot at trade date);
    // only the mark-to-market price per unit moves under a scenario.
    double callUnits = CALL_NOTIONAL / market.spot();
    double putUnits = PUT_NOTIONAL / market.spot();

    List<Scenario> macro = List.of(
        Scenario.of("baseline"),
        Scenario.of("adverse",
            Shock.relative("spot", -0.20), Shock.additive("vol", 0.05), Shock.additive("rate", -0.0075)),
        Scenario.of("severely-adverse",
            Shock.relative("spot", -0.40), Shock.additive("vol", 0.15), Shock.additive("rate", -0.015)));

    List<Scenario> horizons = List.of(
        Scenario.of("Q0", Shock.additive("maturity", 0.00)),
        Scenario.of("Q1", Shock.additive("maturity", -0.25)),
        Scenario.of("Q2", Shock.additive("maturity", -0.50)),
        Scenario.of("Q3", Shock.additive("maturity", -0.75)));

    ScenarioSet grid = crossProduct(macro, horizons);

    System.out.printf(Locale.ROOT, "Stress test showcase: long call / short put book, %d macro scenarios x "
        + "%d horizons = %d full-revaluation nodes%n", macro.size(), horizons.size(), grid.size());
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

      System.out.printf(Locale.ROOT, "%-18s", "loss vs. baseline");
      for (Scenario h : horizons) {
        System.out.printf(Locale.ROOT, "%12s", h.name());
      }
      System.out.println();

      String worstName = null;
      double worstLoss = Double.POSITIVE_INFINITY;
      for (Scenario m : macro) {
        System.out.printf(Locale.ROOT, "%-18s", m.name());
        for (Scenario h : horizons) {
          String name = m.name() + " / " + h.name();
          String baselineName = "baseline / " + h.name();
          double pv = portfolioPv(callUnits, callPv.get(name), putUnits, putPv.get(name));
          double basePv = portfolioPv(callUnits, callPv.get(baselineName), putUnits, putPv.get(baselineName));
          double loss = pv - basePv;
          System.out.printf(Locale.ROOT, "%12.0f", loss);
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

      System.out.println("Every node above is the same recorded kernel, replayed under a moved market —");
      System.out.println("no re-recording between scenarios or horizons. Equal seeds across every replay");
      System.out.println("keep the projected losses comparable (common random numbers). The delta above");
      System.out.println("came from the same grid replay, not a second risk run.");
      System.out.println("What this does not cover: the credit-loss (PD/LGD), PPNR and macro-scenario-");
      System.out.println("generation models a real supervisory stress test also runs — those sit outside");
      System.out.println("a pricing engine.");
    }
  }

  private static double portfolioPv(double callUnits, Nabla.TypedValuation<EquityMarket> call,
                                    double putUnits, Nabla.TypedValuation<EquityMarket> put) {
    return callUnits * call.price() - putUnits * put.price();
  }

  private static double totalSeconds(Map<String, Nabla.TypedValuation<EquityMarket>> results) {
    return results.values().stream().mapToDouble(Nabla.TypedValuation::seconds).sum();
  }

  /**
   * The macro-scenario axis and the horizon axis each bundle several shocks
   * per point, so neither fits a single-input {@code Ladder} — this merges
   * two named {@link Scenario} lists the same way {@link ScenarioSet#grid}
   * merges ladders, just for lists instead.
   */
  private static ScenarioSet crossProduct(List<Scenario> macro, List<Scenario> horizons) {
    List<Scenario> out = new ArrayList<>(macro.size() * horizons.size());
    for (Scenario m : macro) {
      for (Scenario h : horizons) {
        List<Shock> merged = new ArrayList<>(m.shocks());
        merged.addAll(h.shocks());
        out.add(new Scenario(m.name() + " / " + h.name(), merged));
      }
    }
    return new ScenarioSet(out);
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
