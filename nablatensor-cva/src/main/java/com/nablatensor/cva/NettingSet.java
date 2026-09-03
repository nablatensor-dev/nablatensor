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
package com.nablatensor.cva;

import java.util.List;

/**
 * A set of trades with one counterparty under a single master agreement, with an
 * optional CSA. Exposure and CVA are computed at this level — the trades net.
 *
 * @param id           netting-set id
 * @param counterparty the credit name whose default the CVA prices
 * @param trades       the trades in the set (at least one)
 * @param collateral   the CSA, or {@link CollateralAgreement#uncollateralised()}
 */
public record NettingSet(String id, CreditName counterparty, List<CvaTrade> trades,
                         CollateralAgreement collateral) {

  public NettingSet {
    if (trades.isEmpty()) {
      throw new IllegalArgumentException("a netting set needs at least one trade");
    }
    trades = List.copyOf(trades);
  }

  public NettingSet(String id, CreditName counterparty, List<CvaTrade> trades) {
    this(id, counterparty, trades, CollateralAgreement.uncollateralised());
  }

  public double grossNotional() {
    return trades.stream().mapToDouble(CvaTrade::grossNotional).sum();
  }

  /** The latest cash-flow date across the trades — the simulation horizon. */
  public double horizonYears() {
    return trades.stream().mapToDouble(CvaTrade::effectiveMaturityYears).max().orElse(0.0);
  }

  /**
   * Notional-weighted effective maturity, the {@code M_c} the BA-CVA supervisory
   * discount factor and risk weight are applied to.
   */
  public double effectiveMaturityYears() {
    double weighted = 0.0;
    double weight = 0.0;
    for (CvaTrade trade : trades) {
      weighted += trade.grossNotional() * trade.effectiveMaturityYears();
      weight += trade.grossNotional();
    }
    return weight == 0.0 ? horizonYears() : weighted / weight;
  }
}
