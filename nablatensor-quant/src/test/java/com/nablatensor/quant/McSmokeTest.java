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
import com.nablatensor.quant.estimate.Fit;
import com.nablatensor.quant.estimate.Garch11;
import java.util.function.BiConsumer;
import org.junit.jupiter.api.Test;

/**
 * The fast Monte-Carlo smoke: one small, loose-tolerance run of every model path
 * that the {@code @Tag("mc")} validation classes cover in depth. This stays in
 * the default {@code mvn test}; the full closed-form-vs-Monte-Carlo checks run
 * with {@code -P mc}.
 */
class McSmokeTest {

  private static final long PATHS = 60_000L;

  private static <M extends Record> double price(M market, BiConsumer<AadRecorder, Nabla.Inputs<M>> v,
                                                 int steps, long seed) {
    try (Nabla.TypedPricer<M> p = Nabla.model(market, v).fp64().priceOnly().on("cpu-jit").build()) {
      return p.value().with(market).scenarios(PATHS).seed(seed).run().price();
    }
  }

  @Test
  void mertonAndKouJumpModelsPriceAndCollapseToBlackScholes() {
    double bs = GeneralizedBsm.of(OptionType.CALL, 100, 100, 1.0, 0.03, 0.0, 0.2).price();

    double merton0 = price(new MertonJumpMarket(100, 100, 0.2, 0.03, 1.0, 0.0, -0.1, 0.2),
        MertonJumpModel.european(OptionType.CALL, 1.0, 32), 32, 1L);
    assertEquals(bs, merton0, 0.06 * bs, "Merton lambda=0 ~ BSM");

    double kou0 = price(new KouMarket(100, 100, 0.2, 0.03, 1.0, 0.0, 0.4, 10.0, 5.0),
        KouJumpModel.european(OptionType.CALL, 1.0, 32), 32, 2L);
    assertEquals(bs, kou0, 0.06 * bs, "Kou lambda=0 ~ BSM");

    double merton = price(new MertonJumpMarket(100, 100, 0.16, 0.03, 1.0, 1.2, -0.03, 0.2),
        MertonJumpModel.european(OptionType.CALL, 1.0, 32), 32, 3L);
    assertTrue(merton > 0.0 && merton < 40.0, "Merton with jumps prices sanely: " + merton);
  }

  @Test
  void schwartzFuturesAndSpreadOptionPriceSanely() {
    SchwartzMarket sch = new SchwartzMarket(50.0, 1.4, Math.log(55.0), 0.28, 0.03);
    double eST = price(sch, SchwartzOneFactor.european(OptionType.CALL, 0.0, 1.0, 48), 48, 5L)
        * Math.exp(sch.rate() * 1.0);
    assertEquals(SchwartzOneFactor.futuresPrice(sch, 1.0), eST, 0.03 * eST, "Schwartz E[S_T] ~ futures");

    SpreadMarket sp = new SpreadMarket(60, 45, 0.32, 0.28, 0.0, 0.0, 0.03);
    double spread = price(sp, SpreadProducts.spreadOption(6.0, 0.55, 0.5, 32), 32, 7L);
    double kirk = com.nablatensor.quant.analytic.KirkSpreadOption.price(
        60, 45, 6.0, 0.32, 0.28, 0.55, 0.03, 0.0, 0.0, 0.5);
    assertEquals(kirk, spread, 0.08 * kirk + 0.05, "spread MC ~ Kirk");
  }

  @Test
  void bermudanLsmProducesALowerBoundWithGreeks() {
    EquityMarket m = new EquityMarket(40, 40, 0.20, 0.06, 1.0);
    BermudanLsm.Result r = BermudanLsm.price(m, OptionType.PUT, 8, 4, 2, 0.6, 30_000L, 42L);
    assertTrue(r.price() >= r.europeanFloor() - 1e-6, "Bermudan >= European floor");
    assertTrue(r.greeks().spot() > -1.0 && r.greeks().spot() < 0.0, "put delta in (-1, 0)");
  }

  @Test
  void garchFitReturnsAStationaryModel() {
    java.util.Random rng = new java.util.Random(11L);
    double[] ret = new double[1500];
    double s2 = 1e-4;
    for (int i = 0; i < ret.length; i++) {
      ret[i] = Math.sqrt(s2) * rng.nextGaussian();
      s2 = 2e-6 + 0.08 * ret[i] * ret[i] + 0.90 * s2;
    }
    Fit fit = Garch11.fit(ret);
    assertTrue(fit.persistence() > 0.0 && fit.persistence() < 1.0, "stationary fit");
    assertTrue(fit.params().longRunVariance() > 0.0, "positive long-run variance");
  }
}
