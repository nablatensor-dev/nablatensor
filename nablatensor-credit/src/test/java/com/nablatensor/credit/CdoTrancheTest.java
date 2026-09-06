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
package com.nablatensor.credit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.engine.Nabla;
import org.junit.jupiter.api.Test;

/**
 * Feature F9: the loss recursion matches the Vasicek large-pool limit, tranches
 * partition the portfolio loss, correlation moves equity and senior tranches in
 * opposite directions, and the recorded copula Monte-Carlo agrees with the
 * recursion and delivers a correlation delta from one adjoint sweep.
 */
class CdoTrancheTest {

  @Test
  void conditionalDefaultProbabilityIntegratesBackToTheUnconditional() {
    double pd = 0.06;
    double rho = 0.35;
    double[] x = new double[80];
    double[] w = new double[80];
    PortfolioLossDistribution.gaussHermite(80, x, w);
    double norm = 1.0 / Math.sqrt(Math.PI);
    double integral = 0.0;
    for (int g = 0; g < 80; g++) {
      double m = Math.sqrt(2.0) * x[g];
      integral += w[g] * norm * OneFactorGaussianCopula.conditionalDefaultProbability(pd, rho, m);
    }
    assertEquals(pd, integral, 1e-9, "E_M[ p(M) ] = PD");
  }

  @Test
  void lossRecursionMatchesTheVasicekLimit() {
    double pd = 0.04;
    double rho = 0.25;
    double lgd = 0.6;
    PortfolioLossDistribution dist = PortfolioLossDistribution.homogeneous(pd, 1000, rho, lgd, 96);

    assertEquals(pd * lgd, dist.expectedLoss(), 1e-3, "large-pool expected loss = PD * LGD");

    // A tranche's expected loss from the recursion vs the Vasicek CDF.
    CdoTranche mezz = new CdoTranche(0.03, 0.07);
    double recursionEl = dist.expectedTrancheLoss(0.03, 0.07);
    double vasicekEl = vasicekTrancheLoss(0.03, 0.07, pd, rho, lgd);
    assertEquals(vasicekEl, recursionEl, 0.1 * vasicekEl + 1e-4, "mezz EL: recursion vs Vasicek");
    assertTrue(mezz.expectedLossFraction(dist) > 0);
  }

  @Test
  void tranchesPartitionThePortfolioLoss() {
    PortfolioLossDistribution dist = PortfolioLossDistribution.homogeneous(0.05, 100, 0.3, 0.6, 64);
    double whole = dist.expectedTrancheLoss(0.0, 1.0);
    double parts = dist.expectedTrancheLoss(0.0, 0.03)
        + dist.expectedTrancheLoss(0.03, 0.07)
        + dist.expectedTrancheLoss(0.07, 0.15)
        + dist.expectedTrancheLoss(0.15, 1.0);
    assertEquals(dist.expectedLoss(), whole, 1e-12, "0-100% tranche EL = portfolio EL");
    assertEquals(whole, parts, 1e-12, "tranches sum to the whole");
  }

  @Test
  void correlationMovesEquityAndSeniorTranchesOppositeWays() {
    double pd = 0.05;
    double lgd = 0.6;
    int n = 200;
    PortfolioLossDistribution low = PortfolioLossDistribution.homogeneous(pd, n, 0.10, lgd, 96);
    PortfolioLossDistribution high = PortfolioLossDistribution.homogeneous(pd, n, 0.45, lgd, 96);

    double equityLow = low.expectedTrancheLoss(0.0, 0.03);
    double equityHigh = high.expectedTrancheLoss(0.0, 0.03);
    double seniorLow = low.expectedTrancheLoss(0.15, 1.0);
    double seniorHigh = high.expectedTrancheLoss(0.15, 1.0);

    assertTrue(equityHigh < equityLow, "equity tranche EL falls with correlation");
    assertTrue(seniorHigh > seniorLow, "senior tranche EL rises with correlation");
  }

  @Test
  void parSpreadGivesAZeroValueContract() {
    PortfolioLossDistribution dist = PortfolioLossDistribution.homogeneous(0.04, 125, 0.28, 0.6, 96);
    CdoTranche mezz = new CdoTranche(0.03, 0.06);

    double[] times = {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0};
    // With a single terminal loss distribution, approximate the loss path by
    // scaling the terminal expected tranche loss linearly in time.
    double terminalTrancheLoss = dist.expectedTrancheLoss(0.03, 0.06);
    double[] loss = new double[times.length];
    for (int i = 0; i < times.length; i++) {
      loss[i] = terminalTrancheLoss * times[i] / times[times.length - 1];
    }
    double r = 0.03;
    java.util.function.DoubleUnaryOperator df = t -> Math.exp(-r * t);

    double par = mezz.parSpread(loss, times, df);
    assertTrue(par > 0, "positive par spread");
    assertEquals(0.0, mezz.protectionBuyerPv(par, loss, times, df), 1e-10, "par contract has zero PV");
  }

  @Test
  void copulaMonteCarloMatchesTheRecursionAndGivesCorrelationDelta() {
    double pd = 0.05;
    double rho = 0.30;
    double lgd = 0.6;
    int n = 60;
    double t = 1.0;
    double r = 0.02;
    CopulaMarket m = new CopulaMarket(rho, pd);

    var equity = CopulaMonteCarlo.trancheLoss(0.0, 0.04, n, lgd, t, r, 5e-3);

    double disc = Math.exp(-r * t);
    double recursionEl = PortfolioLossDistribution.homogeneous(pd, n, rho, lgd, 96)
        .expectedTrancheLoss(0.0, 0.04);

    try (Nabla.TypedPricer<CopulaMarket> p = Nabla.model(m, equity).fp64().greeks().on("cpu-jit").build()) {
      Nabla.TypedValuation<CopulaMarket> v = p.value().with(m).scenarios(600_000L).seed(42L).run();
      double mcEl = v.price() / disc;
      assertEquals(recursionEl, mcEl, 0.06 * recursionEl + 1e-4, "copula MC vs recursion (equity tranche EL)");

      double adjointDRho = v.greek(CopulaMarket::rho);
      // central bump on the price-only kernel, same seed
      double h = 1e-3;
      double up = price(new CopulaMarket(rho + h, pd), equity);
      double dn = price(new CopulaMarket(rho - h, pd), equity);
      double bump = (up - dn) / (2 * h);
      assertEquals(bump, adjointDRho, 0.05 * Math.abs(bump) + 1e-4, "correlation delta: adjoint vs bump");
      assertTrue(adjointDRho < 0, "equity tranche loss falls with correlation");
    }
  }

  private static double price(CopulaMarket m,
      java.util.function.BiConsumer<com.nablatensor.engine.AadRecorder, Nabla.Inputs<CopulaMarket>> v) {
    try (Nabla.TypedPricer<CopulaMarket> p = Nabla.model(m, v).fp64().priceOnly().on("cpu-jit").build()) {
      return p.value().with(m).scenarios(600_000L).seed(42L).run().price();
    }
  }

  /** Vasicek-limit expected tranche loss by numerically integrating the loss CDF. */
  private static double vasicekTrancheLoss(double a, double d, double pd, double rho, double lgd) {
    int n = 20000;
    double e = 0.0;
    double prev = a;
    for (int i = 1; i <= n; i++) {
      double x = a + (d - a) * i / n;
      double surv = 1.0 - OneFactorGaussianCopula.vasicekLossCdf(0.5 * (prev + x), pd, rho, lgd);
      e += surv * (x - prev);
      prev = x;
    }
    return e;
  }
}
