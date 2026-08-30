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

import com.nablatensor.engine.SDouble;
import org.junit.jupiter.api.Test;

/** Antithetic and control-variate hooks: both must leave the price unbiased, and the
 *  beta=0 control variate must be bit-identical to the raw payoff. */
class HooksTest {

  private static final EquityMarket M = EquityMarket.atmOneYear();
  private static final int STEPS = 60;
  private static final long N = 200_000L;
  private static final long SEED = 999L;

  /** Arithmetic Asian call written against an injected draw source. */
  private static final Hooks.PathPayoff ASIAN = (rec, in, draws, grid) -> {
    SDouble spot = in.of(EquityMarket::spot);
    SDouble rate = in.of(EquityMarket::rate);
    SDouble mat = in.of(EquityMarket::maturity);
    SDouble strike = in.of(EquityMarket::strike);
    GbmPath model = new GbmPath(rec, rate, in.of(EquityMarket::vol), grid, mat);
    SDouble path = spot;
    SDouble sum = rec.constant(0.0);
    for (int t = 0; t < grid.steps(); t++) {
      path = model.step(path, draws.next(), t);
      sum = sum.add(path);
    }
    return sum.div((double) grid.steps()).sub(strike).max(0.0).mul(rate.neg().mul(mat).exp());
  };

  /** Discounted terminal spot; analytic mean is S0. Consumes the same draws as ASIAN. */
  private static final Hooks.PathPayoff DISCOUNTED_TERMINAL = (rec, in, draws, grid) -> {
    SDouble rate = in.of(EquityMarket::rate);
    SDouble mat = in.of(EquityMarket::maturity);
    GbmPath model = new GbmPath(rec, rate, in.of(EquityMarket::vol), grid, mat);
    SDouble path = in.of(EquityMarket::spot);
    for (int t = 0; t < grid.steps(); t++) {
      path = model.step(path, draws.next(), t);
    }
    return path.mul(rate.neg().mul(mat).exp());
  };

  private static double price(Product<EquityMarket> p, String engine) {
    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(p).market(M).steps(STEPS).priceOnly().on(engine).build()) {
      return mc.run(N, SEED).price();
    }
  }

  @Test
  void antitheticIsUnbiasedAndBackendStable() {
    double plain = price(Products.asianCall(), "cpu-jit");
    double anti = price(Hooks.antithetic(ASIAN), "cpu-jit");
    assertEquals(plain, anti, 0.05 * (1 + plain), "antithetic price ~ plain Asian price");

    double antiCpu = price(Hooks.antithetic(ASIAN), "cpu");
    assertEquals(anti, antiCpu, 1e-9 * (1 + anti), "antithetic identical on cpu and cpu-jit");
  }

  @Test
  void controlVariateWithZeroBetaIsExactlyThePlainPayoff() {
    double plain = price(Hooks.antithetic(ASIAN), "cpu-jit");   // any payoff; use ASIAN via a wrapper
    double raw = price(wrap(ASIAN), "cpu-jit");
    double cv0 = price(Hooks.controlVariate(ASIAN, DISCOUNTED_TERMINAL, M.spot(), 0.0), "cpu-jit");
    assertEquals(raw, cv0, 1e-9 * (1 + raw), "beta=0 control variate == raw payoff, bit for bit");
    assertTrue(plain > 0);
  }

  @Test
  void controlVariateWithExactMeanStaysUnbiased() {
    double raw = price(wrap(ASIAN), "cpu-jit");
    double cv1 = price(Hooks.controlVariate(ASIAN, DISCOUNTED_TERMINAL, M.spot(), 1.0), "cpu-jit");
    assertEquals(raw, cv1, 0.03 * (1 + raw), "beta=1 control variate leaves the price unbiased");
  }

  /** Adapts a PathPayoff into a plain Product that just records it with fresh draws. */
  private static Product<EquityMarket> wrap(Hooks.PathPayoff p) {
    return (rec, in, grid) -> rec.output(p.value(rec, in, rec::randn, grid));
  }
}
