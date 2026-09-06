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

import java.util.Arrays;
import com.nablatensor.engine.Nabla;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Caps/floors, swaptions, FX, quanto, floating lookback and the Bermudan shell. */
@Tag("mc")
class NewProductsTest {

  @Test
  void fxOptionMatchesGarmanKohlhagen() {
    FxMarket m = new FxMarket(1.10, 1.12, 0.10, 0.03, 0.015);
    double px = Phase1Support.priceAt(m, FxProducts.fxOption(OptionType.CALL, 1.0, 1));
    // Garman-Kohlhagen == Black-Scholes with carry = rd - rf on a forward S0 e^{(rd-rf)T}
    EquityMarket equiv = new EquityMarket(m.spot(), m.strike(), m.volFx(), m.rateDom() - m.rateForeign(), 1.0);
    BlackScholes bs = BlackScholes.of(OptionType.CALL, equiv);
    double gk = bs.price() * Math.exp(-m.rateForeign() * 1.0);   // discount the extra carry back
    assertEquals(gk, px, 3e-3, "FX call vs Garman-Kohlhagen");
  }

  @Test
  void quantoOptionPricesAndDiffs() {
    QuantoMarket m = QuantoMarket.base();
    var val = FxProducts.quantoOption(OptionType.CALL, 1.0, 32, 1.25);
    double[] adj = Phase1Support.adjoint(m, val);
    assertTrue(adj[0] > 0.0, "quanto call price positive");
    String[] names = Phase1Support.names(QuantoMarket.class);
    int assetSpot = Arrays.asList(names).indexOf("assetSpot");
    double fd = Phase1Support.bump(m, val, assetSpot, 0.5);
    assertEquals(fd, adj[assetSpot + 1], 3e-2 * (1 + Math.abs(fd)), "quanto delta: adjoint vs bump");
  }

  @Test
  void floatingLookbackWorthMoreThanVanilla() {
    EquityMarket m = EquityMarket.atmOneYear();
    int steps = 64;
    double lookback = Phase1Support.priceAt(m, (rec, in) -> Products.floatingLookbackCall().record(rec, in, TimeGrid.uniform(steps)));
    double vanilla = Phase1Support.priceAt(m, (rec, in) -> Products.europeanCall().record(rec, in, TimeGrid.uniform(steps)));
    assertTrue(lookback > vanilla && lookback > 0, "floating lookback call >= vanilla call (" + lookback + " vs " + vanilla + ")");
  }

  @Test
  void lmmCapAdjointMatchesBumpAndSwaptionParityHolds() {
    LmmMarket m = LmmMarket.flat3pct();
    var cap = LmmModel.capFloor(OptionType.CALL, 6, 0.03, 100.0);
    double[] adj = Phase1Support.adjoint(m, cap);
    assertTrue(adj[0] > 0.0, "cap price positive");
    int volIdx = Arrays.asList(Phase1Support.names(LmmMarket.class)).indexOf("vol");
    double fd = Phase1Support.bump(m, cap, volIdx, 5e-3);
    assertEquals(fd, adj[volIdx + 1], 6e-2 * (1 + Math.abs(fd)) + 5e-2, "cap vega: adjoint vs bump");

    // payer - receiver == forward swap value (annuity * (swapRate0 - K)); check the sign/relation
    double payer = Phase1Support.priceAt(m, LmmModel.payerSwaption(2.0, 24, 0.03));
    double receiver = Phase1Support.priceAt(m, LmmModel.receiverSwaption(2.0, 24, 0.03));
    // struck at ~ the initial par rate (0.03 flat), so payer ~ receiver
    assertEquals(receiver, payer, 0.25 * (1 + receiver), "ATM payer ~ receiver swaption");
  }

  @Test
  void hullWhiteSwaptionPricesWithPositiveVega() {
    HullWhiteMarket m = HullWhiteMarket.base();
    var val = HullWhite1F.europeanSwaption(1.0, 4, 1.0, 48, 0.03);
    double[] adj = Phase1Support.adjoint(m, val);
    assertTrue(adj[0] > 0.0, "HW swaption price positive");
    int sig = Arrays.asList(Phase1Support.names(HullWhiteMarket.class)).indexOf("sigma");
    assertTrue(adj[sig + 1] > 0.0, "HW swaption vega > 0");
  }

  @Test
  void bermudanShellCollapsesToEuropean() {
    EquityMarket m = new EquityMarket(100, 105, 0.25, 0.04, 1.0);   // ITM-ish put
    int dates = 6;
    Product berm = BermudanOption.option(OptionType.PUT, dates, 8, 1.0,
        BermudanOption.ContinuationValue.EUROPEAN);
    Product euro = Products.europeanPut();

    double bermPx;
    double euroPx;
    try (MonteCarlo<EquityMarket> b = MonteCarlo.of(berm).market(m).steps(dates * 8).priceOnly().on("cpu-jit").build();
         MonteCarlo<EquityMarket> e = MonteCarlo.of(euro).market(m).steps(dates * 8).priceOnly().on("cpu-jit").build()) {
      bermPx = b.run(300_000L, 7L).price();
      euroPx = e.run(300_000L, 7L).price();
    }
    assertEquals(euroPx, bermPx, 0.05 * (1 + euroPx), "EUROPEAN-continuation Bermudan == European put");

    // exercise-when-ITM is a valid stopping rule -> still a positive, finite price
    Product<EquityMarket> eager = BermudanOption.option(OptionType.PUT, dates, 8, 1.0,
        BermudanOption.ContinuationValue.EXERCISE_WHEN_ITM);
    try (MonteCarlo<EquityMarket> x = MonteCarlo.of(eager).market(m).steps(dates * 8).greeks().on("cpu-jit").build()) {
      Nabla.TypedValuation<EquityMarket> p = x.run(300_000L, 7L);
      assertTrue(p.price() > 0.0 && Double.isFinite(p.greek(EquityMarket::spot)), "exercise-when-ITM Bermudan prices and diffs");
    }
  }
}
