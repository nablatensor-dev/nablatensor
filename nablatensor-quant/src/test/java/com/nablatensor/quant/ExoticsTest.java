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

import com.nablatensor.engine.Nabla;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** The smoothed path-dependent payoffs: in/out parity, the shrinking-width limit, digital vs
 *  the Black-Scholes closed form, and adjoint deltas against a bump of the same smoothed payoff. */
@Tag("mc")
class ExoticsTest {

  private static final EquityMarket M = EquityMarket.atmOneYear();
  private static final int STEPS = 64;
  private static final long N = 200_000L;
  private static final long SEED = 12345L;

  private static double price(Product<EquityMarket> p) {
    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(p).market(M).steps(STEPS).priceOnly().on("cpu-jit").build()) {
      return mc.run(N, SEED).price();
    }
  }

  private static Nabla.TypedValuation<EquityMarket> greeks(Product<EquityMarket> p) {
    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(p).market(M).steps(STEPS).greeks().on("cpu-jit").build()) {
      return mc.run(N, SEED);
    }
  }

  @Test
  void knockOutPlusKnockInEqualsVanilla() {
    double out = price(ExoticProducts.barrier(OptionType.CALL, ExoticProducts.Barrier.UP_OUT, 130.0, 1.0));
    double in = price(ExoticProducts.barrier(OptionType.CALL, ExoticProducts.Barrier.UP_IN, 130.0, 1.0));
    double vanilla = price(Products.europeanCall());
    assertEquals(vanilla, out + in, 1e-9 * (1 + vanilla), "UP_OUT + UP_IN == vanilla (same smoothing, same seed)");
    assertTrue(out < vanilla && out > 0, "0 < knock-out < vanilla");
  }

  @Test
  void barrierConvergesAsSmoothingWidthShrinks() {
    double vanilla = price(Products.europeanCall());
    double[] w = {4.0, 2.0, 1.0, 0.5, 0.25};
    double[] p = new double[w.length];
    for (int i = 0; i < w.length; i++) {
      p[i] = price(ExoticProducts.barrier(OptionType.CALL, ExoticProducts.Barrier.UP_OUT, 120.0, w[i]));
      assertTrue(p[i] >= -1e-9 && p[i] <= vanilla + 1e-9,
          "knock-out price is a valid, sub-vanilla number (w=" + w[i] + " -> " + p[i] + ")");
    }
    // successive refinements move less: the sequence is settling on the sharp-barrier limit
    double lastGap = Math.abs(p[p.length - 1] - p[p.length - 2]);
    double firstGap = Math.abs(p[1] - p[0]);
    assertTrue(lastGap < firstGap, "refinement steps shrink (" + firstGap + " -> " + lastGap + ")");
  }

  @Test
  void barrierAdjointDeltaMatchesABumpOfTheSmoothedPayoff() {
    Product<EquityMarket> p = ExoticProducts.barrier(OptionType.CALL, ExoticProducts.Barrier.UP_OUT, 125.0, 1.5);
    double adjDelta = greeks(p).greek(EquityMarket::spot);

    double h = 0.5;
    double up;
    double dn;
    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(p).market(M).steps(STEPS).priceOnly().on("cpu-jit").build()) {
      up = mc.run(M.withSpot(M.spot() + h), N, SEED).price();
      dn = mc.run(M.withSpot(M.spot() - h), N, SEED).price();
    }
    double fdDelta = (up - dn) / (2 * h);
    assertEquals(fdDelta, adjDelta, 5e-3 + 5e-3 * Math.abs(fdDelta), "barrier adjoint delta vs central bump");
  }

  @Test
  void cashDigitalMatchesBlackScholes() {
    Product<EquityMarket> p = ExoticProducts.digitalCash(OptionType.CALL, 1.0, 0.5);
    double mc;
    try (MonteCarlo<EquityMarket> m = MonteCarlo.of(p).market(M).steps(1).priceOnly().on("cpu-jit").build()) {
      mc = m.run(1_000_000L, SEED).price();
    }
    double d2 = (Math.log(M.spot() / M.strike())
        + (M.rate() - 0.5 * M.vol() * M.vol()) * M.maturity()) / (M.vol() * Math.sqrt(M.maturity()));
    double ref = Math.exp(-M.rate() * M.maturity()) * BlackScholes.N(d2);
    assertEquals(ref, mc, 0.03, "cash-or-nothing digital vs e^{-rT} N(d2)");
  }

  @Test
  void cliquetStaysInsideItsGlobalCollar() {
    double notional = 1000.0;
    double gFloor = 0.0;
    double gCap = 0.30;
    Nabla.TypedValuation<EquityMarket> p = greeks(ExoticProducts.cliquet(-0.05, 0.05, gFloor, gCap, notional));
    double disc = Math.exp(-M.rate() * M.maturity());
    assertTrue(p.price() >= gFloor * notional * disc - 1e-6
        && p.price() <= gCap * notional * disc + 1e-6, "cliquet PV inside its collar: " + p.price());
    assertTrue(Double.isFinite(p.greek(EquityMarket::vol)), "cliquet vega is finite");
  }

  @Test
  void autocallablePricesAndDiffs() {
    Product<EquityMarket> p = ExoticProducts.autocallable(105.0, 0.02, 4, 1.0, 100.0);
    Nabla.TypedValuation<EquityMarket> g = greeks(p);
    assertTrue(g.price() > 0.0, "autocallable PV positive");

    double h = 0.5;
    double up;
    double dn;
    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(p).market(M).steps(STEPS).priceOnly().on("cpu-jit").build()) {
      up = mc.run(M.withSpot(M.spot() + h), N, SEED).price();
      dn = mc.run(M.withSpot(M.spot() - h), N, SEED).price();
    }
    assertEquals((up - dn) / (2 * h), g.greek(EquityMarket::spot), 1e-2 + 1e-2 * Math.abs(g.greek(EquityMarket::spot)), "autocallable adjoint delta vs bump");
  }

  @Test
  void multiMetricEmitsPricePlusThreeNamedRiskMeasures() {
    try (MultiMetric mm = MultiMetric.market(M).steps(STEPS)
        .metric("call", Products.europeanCall())
        .metric("digital", ExoticProducts.digitalCash(OptionType.CALL, 1.0, 1.0))
        .metric("barrierUO", ExoticProducts.barrier(OptionType.CALL, ExoticProducts.Barrier.UP_OUT, 130.0, 1.0))
        .metric("asian", Products.asianCall())
        .on("cpu-jit").build()) {

      var r = mm.run(N, SEED);
      assertEquals(4, r.size());
      for (var e : r.entrySet()) {
        assertTrue(e.getValue().price() > 0.0, e.getKey() + " price positive");
        assertTrue(Double.isFinite(e.getValue().greek(EquityMarket::spot)), e.getKey() + " has a gradient");
      }
      // the "call" metric must match a standalone kernel at the same seed
      double standalone;
      try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.europeanCall()).market(M).steps(STEPS)
          .greeks().on("cpu-jit").build()) {
        standalone = mc.run(N, SEED).price();
      }
      assertEquals(standalone, r.get("call").price(), 1e-9 * (1 + standalone), "metric consistent with standalone");
    }
  }
}
