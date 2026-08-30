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
package com.nablatensor.quant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** The correlated basket and the curve bootstrap + its analytic Jacobian. */
class BasketAndCurveTest {

  // ---- basket ------------------------------------------------------------

  @Test
  void singleNameBasketMatchesBlackScholes() {
    BasketMarket m = new BasketMarket(100, 100, 100, 0.22, 0.30, 0.28, 0.03);
    double[] w = {1.0, 0.0, 0.0};                         // all weight on asset 1
    double[][] corr = {{1, 0.3, 0.2}, {0.3, 1, 0.4}, {0.2, 0.4, 1}};
    double px;
    try (var pricer = com.nablatensor.engine.Nabla.model(m,
        BasketOption.option(OptionType.CALL, w, 100.0, corr, 1.0, 32))
        .fp64().priceOnly().on("cpu-jit").build()) {
      px = pricer.value().with(m).scenarios(600_000L).seed(7L).run().price();
    }
    BlackScholes bs = BlackScholes.of(OptionType.CALL, new EquityMarket(100, 100, 0.22, 0.03, 1.0));
    assertEquals(bs.price(), px, 0.08, "weight-{1,0,0} basket == single-name Black-Scholes");
  }

  @Test
  void basketDeltasMatchBumpAndRisePriceWithCorrelation() {
    BasketMarket m = BasketMarket.equalWeighted();
    double[] w = {1.0 / 3, 1.0 / 3, 1.0 / 3};
    double[][] lowCorr = {{1, 0.1, 0.1}, {0.1, 1, 0.1}, {0.1, 0.1, 1}};
    double[][] highCorr = {{1, 0.9, 0.9}, {0.9, 1, 0.9}, {0.9, 0.9, 1}};

    double[] adj = Phase1Support.adjoint(m, BasketOption.option(OptionType.CALL, w, 100.0, lowCorr, 1.0, 24));
    assertTrue(adj[0] > 0, "basket price positive");
    for (int i = 0; i < 3; i++) {
      double fd = Phase1Support.bump(m, BasketOption.option(OptionType.CALL, w, 100.0, lowCorr, 1.0, 24), i, 0.5);
      assertEquals(fd, adj[i + 1], 3e-2 + 3e-2 * Math.abs(fd), "d(price)/d(s" + (i + 1) + ") adjoint vs bump");
    }

    double lo = Phase1Support.priceAt(m, BasketOption.option(OptionType.CALL, w, 100.0, lowCorr, 1.0, 24));
    double hi = Phase1Support.priceAt(m, BasketOption.option(OptionType.CALL, w, 100.0, highCorr, 1.0, 24));
    assertTrue(hi > lo, "a call on the sum is worth more when the names are more correlated (" + hi + " vs " + lo + ")");
  }

  // ---- curve bootstrap + Jacobian -------------------------------------

  private static final double[] PAR = {0.020, 0.023, 0.025, 0.026, 0.027, 0.0275};

  @Test
  void bootstrapRepricesTheInputInstruments() {
    CurveBootstrap b = CurveBootstrap.fromAnnualParSwaps(PAR);
    YieldCurve c = b.curve();
    for (int i = 0; i < PAR.length; i++) {
      double[] times = new double[i + 1];
      double[] tau = new double[i + 1];
      for (int k = 0; k <= i; k++) {
        times[k] = k + 1.0;
        tau[k] = 1.0;
      }
      assertEquals(PAR[i], c.parSwapRate(times, tau), 1e-12, "re-priced par rate at " + (i + 1) + "y");
    }
  }

  @Test
  void analyticJacobianMatchesFiniteDifferenceAndIsLowerTriangular() {
    CurveBootstrap b = CurveBootstrap.fromAnnualParSwaps(PAR);
    double[][] j = b.zeroRateJacobian();
    double[] z0 = b.curve().zeroRates();
    double h = 1e-6;

    for (int i = 0; i < PAR.length; i++) {
      for (int col = 0; col < PAR.length; col++) {
        if (col > i) {
          assertEquals(0.0, j[i][col], 0.0, "J is lower triangular at (" + i + "," + col + ")");
          continue;
        }
        double[] bumped = PAR.clone();
        bumped[col] += h;
        double zUp = CurveBootstrap.fromAnnualParSwaps(bumped).curve().zeroRates()[i];
        bumped[col] -= 2 * h;
        double zDn = CurveBootstrap.fromAnnualParSwaps(bumped).curve().zeroRates()[i];
        double fd = (zUp - zDn) / (2 * h);
        assertEquals(fd, j[i][col], 1e-6 * (1 + Math.abs(fd)) + 1e-9,
            "dz[" + i + "]/dS[" + col + "]: analytic " + j[i][col] + " vs bump " + fd);
      }
    }
    assertTrue(z0.length == PAR.length);
  }

  @Test
  void mixedDepositFraSwapBootstrapAndJacobian() {
    // 3M deposit, a 3x6 and 6x12 FRA, then consecutive 2y..5y par swaps
    double[] q = {0.021, 0.0225, 0.0235, 0.026, 0.0270, 0.0278, 0.0285};
    CurveBootstrap b = build(q);
    YieldCurve c = b.curve();

    assertEquals(1.0 / (1.0 + q[0] * 0.25), c.discountFactor(0.25), 1e-12, "deposit reprices");
    double fwd = (c.discountFactor(0.25) / c.discountFactor(0.5) - 1.0) / 0.25;
    assertEquals(q[1], fwd, 1e-10, "3x6 FRA reprices");
    assertEquals(q[6], c.parSwapRate(new double[] {1, 2, 3, 4, 5}, new double[] {1, 1, 1, 1, 1}), 1e-9,
        "5y par swap reprices");

    double[][] j = b.zeroRateJacobian();
    double[] pillars = b.pillars();
    double hh = 1e-6;
    for (int i = 0; i < q.length; i++) {
      for (int col = 0; col < q.length; col++) {
        double[] up = q.clone();
        up[col] += hh;
        double[] dn = q.clone();
        dn[col] -= hh;
        double fd = col > i ? 0.0
            : (build(up).curve().zeroRates()[i] - build(dn).curve().zeroRates()[i]) / (2 * hh);
        assertEquals(fd, j[i][col], 1e-5 * (1 + Math.abs(fd)) + 1e-9,
            "dz[" + i + "]/dq[" + col + "] (pillar " + pillars[i] + ")");
      }
    }
  }

  private static CurveBootstrap build(double[] q) {
    return CurveBootstrap.builder()
        .deposit(0.25, q[0])
        .fra(0.25, 0.5, q[1])
        .fra(0.5, 1.0, q[2])
        .swap(2.0, q[3])
        .swap(3.0, q[4])
        .swap(4.0, q[5])
        .swap(5.0, q[6])
        .build();
  }
}
