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
import org.junit.jupiter.api.Test;

/**
 * MVP correctness: the recorded valuations price sensibly, the adjoint Greeks
 * agree with an independent reference, and the bytecode kernel reproduces the
 * scalar oracle path-for-path.
 */
class MvpPricingTest {

  private static final long SCENARIOS = 400_000L;
  private static final long SEED = 20260830L;
  private static final int STEPS = 64;

  @Test
  void europeanCallAdjointGreeksMatchBlackScholes() {
    EquityMarket m = EquityMarket.atmOneYear();
    BlackScholes ref = BlackScholes.of(OptionType.CALL, m);

    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.europeanCall())
        .market(m).steps(1).fp64().greeks().on("cpu-jit").build()) {

      Nabla.TypedValuation<EquityMarket> p = mc.run(SCENARIOS, SEED);

      assertEquals(ref.price(), p.price(), 0.15, "price");
      assertEquals(ref.delta(), p.greek(EquityMarket::spot), 0.02, "delta");
      assertEquals(ref.vega(), p.greek(EquityMarket::vol), 0.60, "vega");
      assertEquals(ref.rho(), p.greek(EquityMarket::rate), 0.60, "rho");
      assertEquals(ref.strikeSensitivity(), p.greek(EquityMarket::strike), 0.02, "dV/dK");
    }
  }

  @Test
  void adjointGreeksMatchCentralBumpForAsian() {
    EquityMarket m = EquityMarket.atmOneYear();

    try (MonteCarlo<EquityMarket> greeks = MonteCarlo.of(Products.asianCall())
             .market(m).steps(STEPS).fp64().greeks().on("cpu-jit").build();
         MonteCarlo<EquityMarket> price = MonteCarlo.of(Products.asianCall())
             .market(m).steps(STEPS).fp64().priceOnly().on("cpu-jit").build()) {

      Nabla.TypedValuation<EquityMarket> adjoint = greeks.run(SCENARIOS, SEED);

      double h = 0.01 * m.spot();
      double up = price.run(m.withSpot(m.spot() + h), SCENARIOS, SEED).price();
      double dn = price.run(m.withSpot(m.spot() - h), SCENARIOS, SEED).price();
      double bumpDelta = (up - dn) / (2 * h);

      assertEquals(bumpDelta, adjoint.greek(EquityMarket::spot), 5e-3, "adjoint delta vs central bump");
      assertTrue(adjoint.price() > 0.0, "Asian call price positive");
    }
  }

  @Test
  void bytecodeKernelReproducesScalarOracle() {
    EquityMarket m = EquityMarket.atmOneYear();

    for (Product<EquityMarket> product : java.util.List.of(
        Products.europeanCall(), Products.asianCall(), Products.lookbackCall(), Products.lookbackPut())) {

      try (MonteCarlo<EquityMarket> oracle = MonteCarlo.of(product)
               .market(m).steps(STEPS).fp64().greeks().on("cpu").build();
           MonteCarlo<EquityMarket> jit = MonteCarlo.of(product)
               .market(m).steps(STEPS).fp64().greeks().on("cpu-jit").build()) {

        Nabla.TypedValuation<EquityMarket> a = oracle.run(SCENARIOS, SEED);
        Nabla.TypedValuation<EquityMarket> b = jit.run(SCENARIOS, SEED);

        assertEquals(a.price(), b.price(), 1e-6 * (1 + Math.abs(a.price())),
            product.label() + " price: cpu-jit vs oracle");
        assertEquals(a.greek(EquityMarket::spot), b.greek(EquityMarket::spot), 1e-6 * (1 + Math.abs(a.greek(EquityMarket::spot))),
            product.label() + " delta: cpu-jit vs oracle");
        assertEquals(a.greek(EquityMarket::vol), b.greek(EquityMarket::vol), 1e-6 * (1 + Math.abs(a.greek(EquityMarket::vol))),
            product.label() + " vega: cpu-jit vs oracle");
      }
    }
  }

  @Test
  void setInputLadderReusesOneKernel() {
    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.europeanCall())
        .market(EquityMarket.atmOneYear()).steps(1).fp64().greeks().on("cpu-jit").build()) {

      double last = Double.NEGATIVE_INFINITY;
      for (double spot = 80.0; spot <= 120.0; spot += 10.0) {
        Nabla.TypedValuation<EquityMarket> p = mc.run(mc.market().withSpot(spot), SCENARIOS, SEED);
        assertTrue(p.price() >= last - 1e-6, "call price monotone in spot along the ladder");
        double d = p.greek(EquityMarket::spot);
        assertTrue(d > 0.0 && d < 1.0 + 1e-6, "call delta in (0,1)");
        last = p.price();
      }
    }
  }
}
