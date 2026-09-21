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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Feature F7: the Merton jump-diffusion step block converges to the exact
 * Poisson-series price, both jump models collapse to Black-Scholes as the
 * intensity vanishes and satisfy put-call parity, and their adjoint spot delta
 * matches a central bump.
 */
@Tag("mc")
class JumpModelsTest {

  private static <M> double price(M market, BiConsumer<AadRecorder, Nabla.Inputs<M>> v,
                                                 long scenarios, long seed) {
    try (Nabla.TypedPricer<M> p = Nabla.model(market, v).fp64().priceOnly().on("cpu-jit").build()) {
      return p.value().with(market).scenarios(scenarios).seed(seed).run().price();
    }
  }

  @Test
  void mertonMonteCarloConvergesToThePoissonSeries() {
    MertonJumpMarket m = MertonJumpMarket.of().spot(100).strike(100).vol(0.18).rate(0.03).maturity(1.0).jumpIntensity(0.75).jumpMean(-0.05).jumpVol(0.15).build();
    double mc = price(m, MertonJumpModel.european(OptionTypeEnum.CALL, 1.0, 128), 1_500_000L, 42L);
    double exact = MertonJumpDiffusion.price(OptionTypeEnum.CALL, m.spot(), m.strike(), m.maturity(),
        m.rate(), m.vol(), m.jumpIntensity(), m.jumpMean(), m.jumpVol());
    assertEquals(exact, mc, 0.015 * exact, "Merton MC vs exact series");
  }

  @Test
  void mertonCollapsesToBlackScholesWhenIntensityIsZero() {
    MertonJumpMarket m = MertonJumpMarket.of().spot(100).strike(100).vol(0.2).rate(0.03).maturity(1.0).jumpIntensity(0.0).jumpMean(-0.1).jumpVol(0.2).build();
    double mc = price(m, MertonJumpModel.european(OptionTypeEnum.CALL, 1.0, 64), 800_000L, 7L);
    double bs = GeneralizedBsm.of().type(OptionTypeEnum.CALL).spot(100).strike(100).maturity(1.0).rate(0.03).dividend(0.0).vol(0.2).build().price();
    assertEquals(bs, mc, 0.02 * bs, "lambda = 0 recovers Black-Scholes");
  }

  @Test
  void mertonPutCallParityHolds() {
    MertonJumpMarket m = MertonJumpMarket.of().spot(100).strike(95).vol(0.18).rate(0.03).maturity(1.0).jumpIntensity(1.0).jumpMean(-0.08).jumpVol(0.2).build();
    var call = MertonJumpModel.european(OptionTypeEnum.CALL, 1.0, 128, 5e-5);
    var put = MertonJumpModel.european(OptionTypeEnum.PUT, 1.0, 128, 5e-5);
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
    MertonJumpMarket withJumps = MertonJumpMarket.of().spot(100).strike(100).vol(0.16).rate(0.03).maturity(1.0).jumpIntensity(1.2).jumpMean(-0.02).jumpVol(0.22).build();
    double jumpPrice = price(withJumps, MertonJumpModel.european(OptionTypeEnum.CALL, 1.0, 96), 1_200_000L, 5L);
    double bs = GeneralizedBsm.of().type(OptionTypeEnum.CALL).spot(100).strike(100).maturity(1.0).rate(0.03).dividend(0.0).vol(0.16).build().price();
    assertTrue(jumpPrice > bs, "jumps add total variance, so the ATM call is worth more: "
        + jumpPrice + " vs " + bs);
  }

  @Test
  void mertonAdjointSpotDeltaMatchesBump() {
    MertonJumpMarket m = MertonJumpMarket.of().spot(100).strike(100).vol(0.18).rate(0.03).maturity(1.0).jumpIntensity(0.8).jumpMean(-0.05).jumpVol(0.15).build();
    var v = MertonJumpModel.european(OptionTypeEnum.CALL, 1.0, 64);
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
    KouMarket zero = KouMarket.of().spot(100).strike(100).vol(0.2).rate(0.03).maturity(1.0).jumpIntensity(0.0).probUp(0.4).etaUp(10.0).etaDown(5.0).build();
    double mc = price(zero, KouJumpModel.european(OptionTypeEnum.CALL, 1.0, 64), 800_000L, 3L);
    double bs = GeneralizedBsm.of().type(OptionTypeEnum.CALL).spot(100).strike(100).maturity(1.0).rate(0.03).dividend(0.0).vol(0.2).build().price();
    assertEquals(bs, mc, 0.02 * bs, "Kou lambda = 0 recovers Black-Scholes");

    KouMarket m = KouMarket.of().spot(100).strike(98).vol(0.16).rate(0.03).maturity(1.0).jumpIntensity(1.0).probUp(0.35).etaUp(12.0).etaDown(6.0).build();
    double call = price(m, KouJumpModel.european(OptionTypeEnum.CALL, 1.0, 96), 1_000_000L, 9L);
    double put = price(m, KouJumpModel.european(OptionTypeEnum.PUT, 1.0, 96), 1_000_000L, 9L);
    assertEquals(m.spot() - m.strike() * Math.exp(-m.rate() * m.maturity()), call - put,
        0.03, "Kou put-call parity");
  }

  @Test
  void kouAdjointSpotDeltaMatchesBump() {
    KouMarket m = KouMarket.of().spot(100).strike(100).vol(0.16).rate(0.03).maturity(1.0).jumpIntensity(0.8).probUp(0.4).etaUp(10.0).etaDown(5.0).build();
    var v = KouJumpModel.european(OptionTypeEnum.CALL, 1.0, 64);
    String[] names = Phase1Support.names(KouMarket.class);
    double[] adj = Phase1Support.adjoint(m, v);
    int spot = Arrays.asList(names).indexOf("spot");
    double bumpDelta = Phase1Support.bump(m, v, spot, 1.0);
    assertEquals(bumpDelta, adj[spot + 1], 5e-3 * (1 + Math.abs(bumpDelta)), "Kou spot delta adjoint vs bump");
  }
}
