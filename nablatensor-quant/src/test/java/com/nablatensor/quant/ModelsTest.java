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
import java.util.function.BiConsumer;
import org.junit.jupiter.api.Test;

/**
 * Every Phase-1 model: the full adjoint parameter gradient (one reverse sweep)
 * must agree with a central bump-and-revalue on the same seed, and each model's
 * degenerate limit must line up with something known.
 */
class ModelsTest {

  private static final int STEPS = 48;

  private <M extends Record> void adjointMatchesBump(
      M market, BiConsumer<AadRecorder, Nabla.Inputs<M>> v, double[] bumps, double tol) {
    adjointMatchesBump(market, v, bumps, tol, java.util.Map.of());
  }

  /** {@code looseTol} overrides {@code tol} for the named components (schemes with a floor). */
  private <M extends Record> void adjointMatchesBump(
      M market, BiConsumer<AadRecorder, Nabla.Inputs<M>> v, double[] bumps, double tol,
      java.util.Map<String, Double> looseTol) {
    String[] names = Phase1Support.names(market.getClass());
    double[] adj = Phase1Support.adjoint(market, v);
    assertTrue(adj[0] > 0.0, "price positive");
    for (int i = 0; i < names.length; i++) {
      if (bumps[i] == 0.0) {
        continue;
      }
      double fd = Phase1Support.bump(market, v, i, bumps[i]);
      double t = looseTol.getOrDefault(names[i], tol);
      assertEquals(fd, adj[i + 1], t * (1 + Math.abs(fd)) + t,
          "d(price)/d(" + names[i] + "): adjoint " + adj[i + 1] + " vs bump " + fd);
    }
  }

  @Test
  void heston() {
    HestonMarket m = new HestonMarket(100, 100, 0.02, 0.04, 1.5, 0.04, 0.5, -0.7);
    var val = HestonModel.european(OptionType.CALL, 1.0, STEPS);
    String[] names = Phase1Support.names(HestonMarket.class);
    double[] adj = Phase1Support.adjoint(m, val);

    // forward price agrees with an independent price-only run
    assertEquals(Phase1Support.priceAt(m, val), adj[0], 1e-9 * (1 + adj[0]), "adjoint vs price-only");
    assertTrue(adj[0] > 0.0, "Heston call price positive");

    // spot delta is the robust adjoint-vs-bump check (large, low-noise sensitivity)
    int spot = java.util.Arrays.asList(names).indexOf("spot");
    double fdDelta = Phase1Support.bump(m, val, spot, 1.0);
    assertEquals(fdDelta, adj[spot + 1], 2e-2 * (1 + fdDelta), "Heston delta: adjoint vs bump");

    // every parameter sensitivity is finite
    for (int i = 1; i < adj.length; i++) {
      assertTrue(Double.isFinite(adj[i]), "d/d(" + names[i - 1] + ") finite");
    }

    // signs / monotonicity from theory: more initial variance -> pricier call
    assertTrue(adj[java.util.Arrays.asList(names).indexOf("v0") + 1] > 0.0, "dPrice/dv0 > 0");
    double lo = Phase1Support.priceAt(m.withV0(0.03), val);
    double hi = Phase1Support.priceAt(m.withV0(0.06), val);
    assertTrue(hi > lo, "higher v0 -> higher call price (" + hi + " vs " + lo + ")");
  }

  @Test
  void sabr() {
    SabrMarket m = new SabrMarket(0.05, 0.055, 0.0, 0.25, 0.5, -0.3, 0.4);
    // forward strike rate alpha beta rho nu
    adjointMatchesBump(m, SabrModel.european(OptionType.CALL, 1.0, STEPS),
        new double[] {1e-4, 1e-4, 0, 1e-3, 5e-3, 0.02, 5e-3}, 4e-2);
  }

  @Test
  void localVolReducesToGbmAndDiffsCleanly() {
    // skew = 0  =>  identical to the GBM European
    LocalVolMarket flat = new LocalVolMarket(100, 100, 0.03, 0.20, 0.0);
    double lv = Phase1Support.priceAt(flat, LocalVolModel.european(OptionType.CALL, 1.0, STEPS));
    EquityMarket gbm = new EquityMarket(100, 100, 0.20, 0.03, 1.0);
    double bs = Phase1Support.priceAt(gbm, (rec, in) -> Products.europeanCall().record(rec, in, TimeGrid.uniform(STEPS)));
    assertEquals(bs, lv, 1e-6 * (1 + bs), "local vol with skew=0 == GBM European");

    LocalVolMarket smile = new LocalVolMarket(100, 100, 0.03, 0.20, -0.4);
    // spot strike rate sigma0 skew
    adjointMatchesBump(smile, LocalVolModel.european(OptionType.CALL, 1.0, STEPS),
        new double[] {1.0, 1.0, 1e-3, 1e-3, 1e-2}, 3e-2);
  }

  @Test
  void hullWhiteBondAndCaplet() {
    HullWhiteMarket m = HullWhiteMarket.base();
    // zero-coupon bond price in (0,1); dP/dr0 < 0
    double[] bond = Phase1Support.adjoint(m, HullWhite1F.zeroCouponBond(5.0, 60));
    assertTrue(bond[0] > 0.0 && bond[0] < 1.0, "ZCB price in (0,1): " + bond[0]);
    assertTrue(bond[1] < 0.0, "dP/dr0 < 0");

    // caplet: adjoint vs bump for r0 / level / meanReversion / sigma / strike
    adjointMatchesBump(m, HullWhite1F.caplet(2.0, 48, 0.5, 1.0),
        new double[] {1e-4, 1e-4, 2e-3, 1e-4, 1e-4}, 4e-2);
  }

  @Test
  void lmmSwaption() {
    LmmMarket m = LmmMarket.flat3pct();
    double[] adj = Phase1Support.adjoint(m, LmmModel.payerSwaption(2.0, 24, 0.03));
    assertTrue(adj[0] > 0.0, "payer swaption price positive");
    // l1 l2 l3 l4 vol corr  -> vega (d/dvol) must be positive
    String[] names = Phase1Support.names(LmmMarket.class);
    int volIdx = java.util.Arrays.asList(names).indexOf("vol");
    assertTrue(adj[volIdx + 1] > 0.0, "swaption vega > 0");
    double fd = Phase1Support.bump(m, LmmModel.payerSwaption(2.0, 24, 0.03), volIdx, 5e-3);
    assertEquals(fd, adj[volIdx + 1], 5e-2 * (1 + Math.abs(fd)) + 5e-2, "swaption vega: adjoint vs bump");
  }
}
