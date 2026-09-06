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
import com.nablatensor.quant.analytic.GeneralizedBsm;
import com.nablatensor.quant.analytic.MertonJumpDiffusion;
import java.util.Arrays;
import java.util.function.BiConsumer;
import org.junit.jupiter.api.Test;

/**
 * Feature F7: the Merton jump-diffusion step block converges to the exact
 * Poisson-series price, both jump models collapse to Black-Scholes as the
 * intensity vanishes and satisfy put-call parity, and their adjoint spot delta
 * matches a central bump.
 */
class JumpModelsTest {

  private static <M extends Record> double price(M market, BiConsumer<AadRecorder, Nabla.Inputs<M>> v,
                                                 long scenarios, long seed) {
    try (Nabla.TypedPricer<M> p = Nabla.model(market, v).fp64().priceOnly().on("cpu-jit").build()) {
      return p.value().with(market).scenarios(scenarios).seed(seed).run().price();
    }
  }

  @Test
  void mertonMonteCarloConvergesToThePoissonSeries() {
    MertonJumpMarket m = new MertonJumpMarket(100, 100, 0.18, 0.03, 1.0, 0.75, -0.05, 0.15);
    double mc = price(m, MertonJumpModel.european(OptionType.CALL, 1.0, 128), 1_500_000L, 42L);
    double exact = MertonJumpDiffusion.price(OptionType.CALL, m.spot(), m.strike(), m.maturity(),
        m.rate(), m.vol(), m.jumpIntensity(), m.jumpMean(), m.jumpVol());
    assertEquals(exact, mc, 0.015 * exact, "Merton MC vs exact series");
  }

  @Test
  void mertonCollapsesToBlackScholesWhenIntensityIsZero() {
    MertonJumpMarket m = new MertonJumpMarket(100, 100, 0.2, 0.03, 1.0, 0.0, -0.1, 0.2);
    double mc = price(m, MertonJumpModel.european(OptionType.CALL, 1.0, 64), 800_000L, 7L);
    double bs = GeneralizedBsm.of(OptionType.CALL, 100, 100, 1.0, 0.03, 0.0, 0.2).price();
    assertEquals(bs, mc, 0.02 * bs, "lambda = 0 recovers Black-Scholes");
  }

  @Test
  void mertonPutCallParityHolds() {
    MertonJumpMarket m = new MertonJumpMarket(100, 95, 0.18, 0.03, 1.0, 1.0, -0.08, 0.2);
    var call = MertonJumpModel.european(OptionType.CALL, 1.0, 128, 5e-5);
    var put = MertonJumpModel.european(OptionType.PUT, 1.0, 128, 5e-5);
    double c = price(m, call, 2_000_000L, 11L);
    double p = price(m, put, 2_000_000L, 11L);
    // CRN: c - p = e^{-rT}(mean(S_T) - K), so this checks E[S_T] = S0 e^{rT}.
    // The smoothed jump indicator leaves an O(0.1%) martingale bias; 0.05 on a
    // price of ~7.8 is comfortably inside that.
    assertEquals(m.spot() - m.strike() * Math.exp(-m.rate() * m.maturity()), c - p,
        0.05, "Merton put-call parity");
  }

  @Test
  void mertonJumpsAddValueToAnAtmCall() {
    MertonJumpMarket withJumps = new MertonJumpMarket(100, 100, 0.16, 0.03, 1.0, 1.2, -0.02, 0.22);
    double jumpPrice = price(withJumps, MertonJumpModel.european(OptionType.CALL, 1.0, 96), 1_200_000L, 5L);
    double bs = GeneralizedBsm.of(OptionType.CALL, 100, 100, 1.0, 0.03, 0.0, 0.16).price();
    assertTrue(jumpPrice > bs, "jumps add total variance, so the ATM call is worth more: "
        + jumpPrice + " vs " + bs);
  }

  @Test
  void mertonAdjointSpotDeltaMatchesBump() {
    MertonJumpMarket m = new MertonJumpMarket(100, 100, 0.18, 0.03, 1.0, 0.8, -0.05, 0.15);
    var v = MertonJumpModel.european(OptionType.CALL, 1.0, 64);
    String[] names = Phase1Support.names(MertonJumpMarket.class);
    double[] adj = Phase1Support.adjoint(m, v);
    int spot = Arrays.asList(names).indexOf("spot");
    int jm = Arrays.asList(names).indexOf("jumpMean");

    double bumpDelta = Phase1Support.bump(m, v, spot, 1.0);
    assertEquals(bumpDelta, adj[spot + 1], 5e-3 * (1 + Math.abs(bumpDelta)), "spot delta adjoint vs bump");

    double bumpJm = Phase1Support.bump(m, v, jm, 5e-3);
    assertEquals(bumpJm, adj[jm + 1], 0.08 * (1 + Math.abs(bumpJm)), "d/d(jumpMean) adjoint vs bump");
  }

  @Test
  void kouCollapsesToBlackScholesAndSatisfiesParity() {
    KouMarket zero = new KouMarket(100, 100, 0.2, 0.03, 1.0, 0.0, 0.4, 10.0, 5.0);
    double mc = price(zero, KouJumpModel.european(OptionType.CALL, 1.0, 64), 800_000L, 3L);
    double bs = GeneralizedBsm.of(OptionType.CALL, 100, 100, 1.0, 0.03, 0.0, 0.2).price();
    assertEquals(bs, mc, 0.02 * bs, "Kou lambda = 0 recovers Black-Scholes");

    KouMarket m = new KouMarket(100, 98, 0.16, 0.03, 1.0, 1.0, 0.35, 12.0, 6.0);
    double call = price(m, KouJumpModel.european(OptionType.CALL, 1.0, 96), 1_000_000L, 9L);
    double put = price(m, KouJumpModel.european(OptionType.PUT, 1.0, 96), 1_000_000L, 9L);
    assertEquals(m.spot() - m.strike() * Math.exp(-m.rate() * m.maturity()), call - put,
        0.03, "Kou put-call parity");
  }

  @Test
  void kouAdjointSpotDeltaMatchesBump() {
    KouMarket m = new KouMarket(100, 100, 0.16, 0.03, 1.0, 0.8, 0.4, 10.0, 5.0);
    var v = KouJumpModel.european(OptionType.CALL, 1.0, 64);
    String[] names = Phase1Support.names(KouMarket.class);
    double[] adj = Phase1Support.adjoint(m, v);
    int spot = Arrays.asList(names).indexOf("spot");
    double bumpDelta = Phase1Support.bump(m, v, spot, 1.0);
    assertEquals(bumpDelta, adj[spot + 1], 5e-3 * (1 + Math.abs(bumpDelta)), "Kou spot delta adjoint vs bump");
  }
}
