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
package com.nablatensor.risk;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Products;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Per-trade adjoint sensitivities, aggregated to a netting set / book, reconcile
 * to a full one-factor-at-a-time bump grid on the whole book — the Phase-2
 * definition-of-done for the aggregation layer.
 */
class PortfolioAggregationTest {

  private static final long N = 120_000L;
  private static final long SEED = 314159L;
  private static final int STEPS = 16;

  /** A vanilla-call position on one equity name. */
  private record EqPosition(String id, String nettingSet, String bucket, String name,
                            EquityMarket market, double weight) {

    double price(EquityMarket m) {
      try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.europeanCall()).market(m).steps(STEPS)
          .priceOnly().on("cpu-jit").build()) {
        return weight * mc.run(N, SEED).price();
      }
    }

    Sensitivities adjointSensitivities() {
      try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.europeanCall()).market(market).steps(STEPS)
          .greeks().on("cpu-jit").build()) {
        var p = mc.run(N, SEED);
        return Sensitivities.builder()
            .add(RiskFactor.equityDelta(bucket, name), weight * p.greek(EquityMarket::spot))
            .add(RiskFactor.equityVega(bucket, name, 1.0), weight * p.greek(EquityMarket::vol))
            .build();
      }
    }
  }

  @Test
  void bookDeltaVectorMatchesAOneAtATimeBumpGrid() {
    List<EqPosition> book = List.of(
        new EqPosition("t1", "NS_A", "5", "ACME", new EquityMarket(100, 100, 0.20, 0.03, 1.0), 1.0),
        new EqPosition("t2", "NS_A", "5", "ACME", new EquityMarket(100, 105, 0.22, 0.03, 1.0), -0.5),
        new EqPosition("t3", "NS_B", "6", "GLOBEX", new EquityMarket(50, 48, 0.30, 0.03, 1.0), 2.0),
        new EqPosition("t4", "NS_B", "6", "INITECH", new EquityMarket(75, 80, 0.25, 0.03, 1.0), 1.0));

    Portfolio portfolio = new Portfolio(book.stream()
        .map(p -> Portfolio.trade(p.id(), p.nettingSet(), p.adjointSensitivities()))
        .toList());
    Sensitivities agg = portfolio.aggregate();

    // netting-set split must add back up to the book
    Map<String, Sensitivities> byNs = portfolio.byNettingSet();
    Sensitivities recombined = byNs.get("NS_A").plus(byNs.get("NS_B"));
    for (var f : agg.asMap().keySet()) {
      assertEquals(agg.get(f), recombined.get(f), 1e-9, "netting-set split reconstructs " + f);
    }

    // full bump grid: bump each name's spot, reprice every affected position
    for (String name : List.of("ACME", "GLOBEX", "INITECH")) {
      double base0 = 0;
      double spot = book.stream().filter(p -> p.name().equals(name)).findFirst().orElseThrow().market().spot();
      double h = 0.01 * spot;
      double up = 0;
      double dn = 0;
      for (EqPosition p : book) {
        if (!p.name().equals(name)) {
          continue;
        }
        up += p.price(p.market().withSpot(p.market().spot() + h));
        dn += p.price(p.market().withSpot(p.market().spot() - h));
        base0 += p.price(p.market());
      }
      double bumpDelta = (up - dn) / (2 * h);
      String bucket = name.equals("ACME") ? "5" : "6";
      double adjDelta = agg.get(RiskFactor.equityDelta(bucket, name));
      assertEquals(bumpDelta, adjDelta, 1e-2 * (1 + Math.abs(bumpDelta)) + 5e-3,
          "book delta to " + name + ": aggregate adjoint vs bump grid");
      org.junit.jupiter.api.Assertions.assertTrue(Double.isFinite(base0));
    }
  }
}
