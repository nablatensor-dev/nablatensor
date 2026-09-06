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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskMeasure;
import com.nablatensor.tensor.NablaTensors;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Every example is also a test: each {@code main} must run to completion on a
 * small scenario count. Keeps the docs pages from bit-rotting.
 */
class ExamplesSmokeTest {

  @Test
  void vanillaEuropean() {
    System.setProperty("scenarios", "50000");
    VanillaEuropeanGreeks.main(new String[0]);
  }

  @Test
  void analyticVsAdjoint() {
    System.setProperty("scenarios", "50000");
    AnalyticVsAdjoint.main(new String[0]);
  }

  @Test
  void blackScholesBothWays() {
    System.setProperty("scenarios", "20000");
    System.setProperty("engine", "cpu-jit");
    BlackScholesBothWays.main(new String[0]);
  }

  @Test
  void garchMle() {
    System.setProperty("obs", "3000");
    GarchMleShowcase.main(new String[0]);
  }

  @Test
  void varEs() {
    System.setProperty("window", "3000");
    VarEsShowcase.main(new String[0]);
  }

  @Test
  void multiCurveBootstrap() {
    MultiCurveBootstrapShowcase.main(new String[0]);
  }

  @Test
  void hullWhiteCalibration() {
    HullWhiteCalibrationShowcase.main(new String[0]);
  }

  @Test
  void convexityQuanto() {
    ConvexityQuantoShowcase.main(new String[0]);
  }

  @Test
  void jumpDiffusion() {
    System.setProperty("paths", "40000");
    System.setProperty("steps", "32");
    JumpDiffusionShowcase.main(new String[0]);
  }

  @Test
  void cosCalibration() {
    CosCalibrationShowcase.main(new String[0]);
  }

  @Test
  void sparkSpread() {
    System.setProperty("paths", "40000");
    SparkSpreadShowcase.main(new String[0]);
  }

  @Test
  void cdoTranche() {
    CdoTrancheShowcase.main(new String[0]);
  }

  @Test
  void bermudanLsm() {
    System.setProperty("paths", "20000");
    System.setProperty("dates", "8");
    BermudanLsmShowcase.main(new String[0]);
  }

  @Test
  void latticeConvergence() {
    LatticeConvergenceShowcase.main(new String[0]);
  }

  @Test
  void asianAcrossBackends() {
    System.setProperty("scenarios", "50000");
    System.setProperty("steps", "32");
    AsianGreeksBackends.main(new String[0]);
  }

  @Test
  void frtbCurvature() {
    System.setProperty("scenarios", "10000");
    System.setProperty("steps", "16");
    System.setProperty("engine", "cpu-jit");
    FrtbCurvatureShowcase.main(new String[0]);
  }

  @Test
  void curvatureStripsLinearPnlAndTakesTheWorseShock() {
    assertEquals(11.0,
        FrtbCurvatureShowcase.curvatureValue(-10.0, -18.0, -6.0, 30.0, -0.5), 1e-12);
  }

  @Test
  void frtbFull() {
    System.setProperty("scenarios", "5000");
    System.setProperty("steps", "8");
    System.setProperty("engine", "cpu-jit");
    FrtbFullShowcase.main(new String[0]);
  }

  @Test
  void frtbFullCoversTablesNettingAndCapitalComponents() {
    var parameters = FrtbFullShowcase.ParameterSet.demoMar21();
    assertEquals(7, parameters.tables().size());
    assertEquals(89, parameters.bucketCount());

    var market = new FrtbFullShowcase.MarketData(
        java.time.LocalDate.of(2026, 9, 2),
        Map.of("SPX", 100.0, "USD-OIS-5Y", 0.042, "ACME-SPREAD-5Y", 0.012,
            "EURUSD", 1.085, "WTI-1Y", 76.5, "IMPLIED-VOL", 0.20));
    var trades = List.of(
        new FrtbFullShowcase.TradeSpec("T1", "NS1", 10_000_000, 1),
        new FrtbFullShowcase.TradeSpec("T2", "NS1", 4_000_000, -1),
        new FrtbFullShowcase.TradeSpec("T3", "NS2", 2_000_000, 1));
    var portfolio = FrtbFullShowcase.buildPortfolio(trades, parameters, market, 3.5);
    assertEquals(2, portfolio.byNettingSet().size());
    var netted = portfolio.aggregate();
    int tradeFactorObservations = portfolio.trades().stream()
      .mapToInt(trade -> trade.sensitivities().asMap().size()).sum();
    assertTrue(netted.asMap().size() < tradeFactorObservations);
    var capital = FrtbFullShowcase.aggregate(parameters, netted);

    for (RiskClass riskClass : RiskClass.values()) {
      assertFalse(netted.ofClass(riskClass).ofMeasure(RiskMeasure.DELTA).isEmpty());
      assertFalse(netted.ofClass(riskClass).ofMeasure(RiskMeasure.VEGA).isEmpty());
      assertFalse(netted.ofClass(riskClass).ofMeasure(RiskMeasure.CURVATURE).isEmpty());
      var classCapital = capital.byClass().get(riskClass);
      assertTrue(classCapital.total() > 0.0);
      for (var measure : List.of(
          classCapital.delta(), classCapital.vega(), classCapital.curvature())) {
        assertEquals(CorrelationScenario.values().length, measure.scenarios().size());
        assertEquals(measure.scenarios().values().stream().mapToDouble(Double::doubleValue).max()
            .orElseThrow(), measure.total(), 1e-12);
      }
    }

    var drc = FrtbFullShowcase.defaultRiskCharge(List.of(
        new FrtbFullShowcase.DefaultPosition("A", 1_000_000, 0.03),
        new FrtbFullShowcase.DefaultPosition("B", -200_000, 0.15)));
    var rrao = FrtbFullShowcase.residualRiskAddOn(List.of(
        new FrtbFullShowcase.ResidualPosition("X", 1_000_000, true),
        new FrtbFullShowcase.ResidualPosition("Y", 2_000_000, false)));
    assertTrue(drc.total() > 0.0);
    assertEquals(0.012, rrao.total(), 1e-12);
  }

  @Test
  void swapThePayoff() {
    System.setProperty("scenarios", "50000");
    System.setProperty("steps", "32");
    SwapThePayoff.main(new String[0]);
  }

  @Test
  void sabrCalibration() {
    HestonSabrCalibration.main(new String[0]);
  }

  @Test
  void mnistMlp() throws Exception {
    assumeTrue(!NablaTensors.devices().isEmpty(),
        "MnistMlp needs a tensor compute backend (Vulkan / ROCm / CUDA)");
    System.setProperty("epochs", "5");
    System.setProperty("batch", "64");
    System.setProperty("hidden", "32");
    System.setProperty("mnistCsv", "does-not-exist.csv");  // force the synthetic fallback
    MnistMlp.main(new String[0]);
  }
}
