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

import com.nablatensor.engine.ADouble;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Importance sampling and path filtering: both leave the target expectation intact. */
@Tag("mc")
class HooksExtraTest {

  private static final EquityMarket M = EquityMarket.of().spot(100).strike(130).vol(0.20).rate(0.03).maturity(1.0).build();  // deep OTM call
  private static final int STEPS = 40;
  private static final long N = 400_000L;
  private static final long SEED = 555L;

  /** OTM European call over an injected draw stream. */
  private static final Hooks.PathPayoff CALL = (rec, in, draws, grid) -> {
    ADouble rate = in.of(EquityMarket::rate);
    ADouble mat = in.of(EquityMarket::maturity);
    GbmPath model = GbmPath.of(rec, rate, in.of(EquityMarket::vol), grid, mat);
    ADouble s = in.of(EquityMarket::spot);
    for (int t = 0; t < grid.steps(); t++) {
      s = model.step(s, draws.next(), t);
    }
    return s.sub(in.of(EquityMarket::strike)).max(0.0).mul(rate.neg().mul(mat).exp());
  };

  /** Terminal log-moneyness; positive when the call finishes in the money. */
  private static final Hooks.PathPayoff FINISHES_ITM = (rec, in, draws, grid) -> {
    ADouble rate = in.of(EquityMarket::rate);
    ADouble mat = in.of(EquityMarket::maturity);
    GbmPath model = GbmPath.of(rec, rate, in.of(EquityMarket::vol), grid, mat);
    ADouble s = in.of(EquityMarket::spot);
    for (int t = 0; t < grid.steps(); t++) {
      s = model.step(s, draws.next(), t);
    }
    return s.sub(in.of(EquityMarket::strike));
  };

  private static double price(Product<EquityMarket> p) {
    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(p).market(M).steps(STEPS).priceOnly().on("cpu-jit").build()) {
      return mc.run(N, SEED).price();
    }
  }

  private static Product<EquityMarket> raw(Hooks.PathPayoff p) {
    return (rec, in, grid) -> rec.output(p.value(rec, in, rec::randn, grid));
  }

  @Test
  void importanceSamplingIsUnbiasedForTheOtmCall() {
    double plain = price(raw(CALL));
    double is = price(Hooks.ImportanceSampling.of().payoff(CALL).muPerStep(0.10).build());   // shift paths toward the strike
    assertTrue(plain > 0.0, "plain OTM call price positive");
    assertEquals(plain, is, 0.10 * (1 + plain), "importance-sampled price ~ plain price");
  }

  @Test
  void pathFilterWithAnAlwaysTrueConditionIsThePlainPayoff() {
    double plain = price(raw(CALL));
    // condition >> 0 everywhere -> smoothed indicator ~ 1
    double filtered = price(Hooks.PathFilter.of().payoff(CALL).condition((rec, in, d, grid) -> {
      // consume the same number of draws, then return a large constant
      for (int t = 0; t < grid.steps(); t++) {
        d.next();
      }
      return rec.constant(1e6);
    }).width(1.0).build());
    assertEquals(plain, filtered, 1e-6 * (1 + plain), "filter(always-true) == plain payoff");
  }

  @Test
  void pathFilterOnFinishesItmLeavesTheCallUnchanged() {
    // a call already pays 0 off the ITM set, so multiplying by 1{S_T > K} changes nothing
    double plain = price(raw(CALL));
    double filtered = price(Hooks.PathFilter.of().payoff(CALL).condition(FINISHES_ITM).width(0.5).build());
    assertEquals(plain, filtered, 0.03 * (1 + plain), "filter(finishes-ITM) leaves an OTM call unchanged");
  }
}
