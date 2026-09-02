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
import com.nablatensor.risk.NestedAggregation;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import com.nablatensor.scenario.Scenario;
import com.nablatensor.scenario.ScenarioRunner;
import com.nablatensor.scenario.ScenarioSet;
import com.nablatensor.scenario.Shock;
import java.util.Locale;
import java.util.Map;

/**
 * The expensive part of FRTB curvature: two full shocked repricings of a
 * path-dependent product, with the linear delta P&amp;L removed afterwards.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.FrtbCurvatureShowcase}
 */
public final class FrtbCurvatureShowcase {

  private static final double EQUITY_CURVATURE_RISK_WEIGHT = 0.30;
  private static final double SHORT_POSITION = -1.0;

  private FrtbCurvatureShowcase() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    int steps = Integer.getInteger("steps", 252);
    long scenarios = Long.getLong("scenarios", 1_000_000L);
    long seed = Long.getLong("seed", 42L);
    long bankFactors = Long.getLong("bankFactors", 10_000L);
    String engine = System.getProperty("engine", "cpu-jit");

    ScenarioSet shocks = ScenarioSet.list(
        Scenario.of("base"),
        Scenario.of("spot-up", Shock.relative("spot", EQUITY_CURVATURE_RISK_WEIGHT)),
        Scenario.of("spot-down", Shock.relative("spot", -EQUITY_CURVATURE_RISK_WEIGHT)));

    System.out.printf(Locale.ROOT, "FRTB curvature showcase: short arithmetic Asian call%n");
    System.out.printf(Locale.ROOT, "%,d scenarios x %d fixings, engine %s, seed %d%n%n",
        scenarios, steps, engine, seed);

    try (MonteCarlo<EquityMarket> greeks = MonteCarlo.of(Products.asianCall())
         .market(market).steps(steps).fp64().greeks().on(engine).build();
       MonteCarlo<EquityMarket> pricer = MonteCarlo.of(Products.asianCall())
         .market(market).steps(steps).fp64().on(engine).build()) {
      long warmScenarios = Math.min(scenarios, 50_000L);
      greeks.run(warmScenarios, seed);
      pricer.run(warmScenarios, seed);

      Nabla.TypedValuation<EquityMarket> deltaRun = greeks.run(market, scenarios, seed);
      Map<String, Nabla.TypedValuation<EquityMarket>> valuations =
        ScenarioRunner.run(pricer, market, shocks, scenarios, seed);
      Nabla.TypedValuation<EquityMarket> base = valuations.get("base");
      Nabla.TypedValuation<EquityMarket> up = valuations.get("spot-up");
      Nabla.TypedValuation<EquityMarket> down = valuations.get("spot-down");

      double pvBase = SHORT_POSITION * base.price();
      double pvUp = SHORT_POSITION * up.price();
      double pvDown = SHORT_POSITION * down.price();
      double delta = SHORT_POSITION * deltaRun.greek(EquityMarket::spot);
      double shock = EQUITY_CURVATURE_RISK_WEIGHT * market.spot();
      double cvr = curvatureValue(pvBase, pvUp, pvDown, shock, delta);

      RiskFactor factor = RiskFactor.equityDelta("5", "ASIAN-CALL").asCurvature();
      Sensitivities curvature = Sensitivities.builder().add(factor, cvr).build();
      double charge = NestedAggregation.curvature(
          (left, right) -> left.equals(right) ? 1.0 : 0.25,
          (left, right) -> left.equals(right) ? 1.0 : 0.15)
          .aggregate(curvature).total();

      double repricingSeconds = base.seconds() + up.seconds() + down.seconds();
      double shockedPairSeconds = up.seconds() + down.seconds();
      double projectedSeconds = base.seconds() + bankFactors * shockedPairSeconds;

      System.out.printf(Locale.ROOT, "base PV (short)       %12.6f%n", pvBase);
      System.out.printf(Locale.ROOT, "base delta            %12.6f%n", delta);
      System.out.printf(Locale.ROOT, "spot shock             %12.6f  (+/- %.0f%%)%n",
          shock, 100.0 * EQUITY_CURVATURE_RISK_WEIGHT);
      System.out.printf(Locale.ROOT, "up/down PV             %12.6f / %12.6f%n", pvUp, pvDown);
      System.out.printf(Locale.ROOT, "curvature CVR          %12.6f%n", cvr);
      System.out.printf(Locale.ROOT, "curvature charge       %12.6f%n%n", charge);
      System.out.printf(Locale.ROOT, "adjoint delta run      %12s%n", wall(deltaRun.seconds()));
      System.out.printf(Locale.ROOT, "three price replays    %12s%n", wall(repricingSeconds));
      System.out.printf(Locale.ROOT, "up/down replay pair    %12s%n", wall(shockedPairSeconds));
      System.out.printf(Locale.ROOT, "%,d-factor sequential projection %s%n%n",
          bankFactors, wall(projectedSeconds));
      System.out.println("The heavy operation is the shocked full repricing. CVR arithmetic and");
      System.out.println("NestedAggregation are small scalar post-processing steps.");
      System.out.println("Equal seeds provide common random numbers across base/up/down runs.");
    }
  }

  static double curvatureValue(double pvBase, double pvUp, double pvDown,
                               double shock, double delta) {
    double upNonLinear = pvUp - pvBase - shock * delta;
    double downNonLinear = pvDown - pvBase + shock * delta;
    return -Math.min(upNonLinear, downNonLinear);
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