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

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.analytic.KirkSpreadOption;
import com.nablatensor.quant.analytic.Margrabe;
import java.util.Arrays;
import java.util.function.BiConsumer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Feature F10: Kirk collapses to Margrabe at zero strike, both agree with a
 * correlated-GBM Monte-Carlo, the Schwartz futures price matches its simulation,
 * and the seasonality fit recovers its coefficients.
 */
@Tag("mc")
class CommoditySpreadTest {

  private static <M extends Record> double price(M market, BiConsumer<AadRecorder, Nabla.Inputs<M>> v,
                                                 long scenarios, long seed) {
    try (Nabla.TypedPricer<M> p = Nabla.model(market, v).fp64().priceOnly().on("cpu-jit").build()) {
      return p.value().with(market).scenarios(scenarios).seed(seed).run().price();
    }
  }

  @Test
  void kirkCollapsesToMargrabeAtZeroStrike() {
    double s1 = 62, s2 = 48, v1 = 0.34, v2 = 0.29, rho = 0.55, r = 0.03, q1 = 0.01, q2 = 0.02, t = 1.0;
    double kirk = KirkSpreadOption.price(s1, s2, 0.0, v1, v2, rho, r, q1, q2, t);
    double margrabe = Margrabe.of(s1, s2, v1, v2, rho, q1, q2, t).price();
    assertEquals(margrabe, kirk, 1e-9, "Kirk(K=0) == Margrabe");
  }

  @Test
  void margrabeAgreesWithCorrelatedGbmMonteCarlo() {
    SpreadMarket m = new SpreadMarket(60, 45, 0.32, 0.28, 0.0, 0.0, 0.03);
    double rho = 0.6;
    double mc = price(m, SpreadProducts.spreadOption(0.0, rho, 1.0, 64), 1_500_000L, 42L);
    double margrabe = Margrabe.of(m.s1(), m.s2(), m.vol1(), m.vol2(), rho, 0.0, 0.0, 1.0).price();
    assertEquals(margrabe, mc, 0.02 * margrabe, "exchange option: MC vs Margrabe");
  }

  @Test
  void kirkAgreesWithMonteCarloForANonZeroStrike() {
    SpreadMarket m = new SpreadMarket(60, 45, 0.30, 0.26, 0.005, 0.010, 0.03);
    double rho = 0.5;
    double k = 8.0;
    double mc = price(m, SpreadProducts.spreadOption(k, rho, 1.0, 64), 1_500_000L, 7L);
    double kirk = KirkSpreadOption.price(m.s1(), m.s2(), k, m.vol1(), m.vol2(), rho, m.rate(),
        m.yield1(), m.yield2(), 1.0);
    assertEquals(mc, kirk, 0.03 * mc + 0.02, "spread option: MC vs Kirk approximation");
  }

  @Test
  void spreadMonteCarloAdjointDeltaMatchesBump() {
    SpreadMarket m = new SpreadMarket(58, 46, 0.33, 0.27, 0.0, 0.0, 0.03);
    var v = SpreadProducts.spreadOption(6.0, 0.5, 1.0, 48);
    String[] names = Phase1Support.names(SpreadMarket.class);
    double[] adj = Phase1Support.adjoint(m, v);
    int s1 = Arrays.asList(names).indexOf("s1");
    int s2 = Arrays.asList(names).indexOf("s2");
    assertEquals(Phase1Support.bump(m, v, s1, 0.5), adj[s1 + 1], 5e-3 * (1 + Math.abs(adj[s1 + 1])), "d/dS1");
    assertEquals(Phase1Support.bump(m, v, s2, 0.5), adj[s2 + 1], 5e-3 * (1 + Math.abs(adj[s2 + 1])), "d/dS2");
    assertTrue(adj[s1 + 1] > 0 && adj[s2 + 1] < 0, "long the first leg, short the second");
  }

  @Test
  void schwartzFuturesPriceMatchesSimulation() {
    SchwartzMarket m = new SchwartzMarket(50.0, 1.5, Math.log(58.0), 0.28, 0.03);
    double t = 1.5;
    // E[S_T] = e^{rT} * price of a zero-strike call.
    double eST = price(m, SchwartzOneFactor.european(OptionType.CALL, 0.0, t, 160), 1_000_000L, 11L)
        * Math.exp(m.rate() * t);
    double futures = SchwartzOneFactor.futuresPrice(m, t);
    assertEquals(futures, eST, 0.01 * futures, "Schwartz futures price vs MC E[S_T]");

    // Far horizon: the log price is stationary at mean `level`, variance sigma^2/(2 kappa).
    double farFutures = SchwartzOneFactor.futuresPrice(m, 40.0);
    double stationary = Math.exp(m.level() + m.sigma() * m.sigma() / (4.0 * m.kappa()));
    assertEquals(stationary, farFutures, 1e-6 * stationary, "far-horizon futures -> stationary");
  }

  @Test
  void schwartzAdjointMatchesBump() {
    SchwartzMarket m = new SchwartzMarket(50.0, 1.3, Math.log(55.0), 0.30, 0.03);
    var v = SchwartzOneFactor.european(OptionType.CALL, 52.0, 1.0, 120);
    String[] names = Phase1Support.names(SchwartzMarket.class);
    double[] adj = Phase1Support.adjoint(m, v);
    int spot = Arrays.asList(names).indexOf("spot");
    int level = Arrays.asList(names).indexOf("level");
    assertEquals(Phase1Support.bump(m, v, spot, 0.5), adj[spot + 1], 5e-3 * (1 + Math.abs(adj[spot + 1])),
        "d/d(spot)");
    assertEquals(Phase1Support.bump(m, v, level, 5e-3), adj[level + 1], 0.02 * (1 + Math.abs(adj[level + 1])),
        "d/d(level)");
  }

  @Test
  void seasonalityFitRecoversCoefficients() {
    Seasonality truth = new Seasonality(new double[] {0.08, -0.03}, new double[] {0.05, 0.02});
    int n = 96;
    double[] times = new double[n];
    double[] values = new double[n];
    for (int i = 0; i < n; i++) {
      times[i] = i / 12.0;                    // monthly over 8 years
      values[i] = truth.value(times[i]);
    }
    Seasonality fit = Seasonality.fit(times, values, 2);
    assertEquals(0.08, fit.aCos()[0], 1e-9);
    assertEquals(-0.03, fit.aCos()[1], 1e-9);
    assertEquals(0.05, fit.aSin()[0], 1e-9);
    assertEquals(0.02, fit.aSin()[1], 1e-9);
    // Annual periodicity.
    assertEquals(truth.value(0.3), truth.value(1.3), 1e-12);
  }
}
