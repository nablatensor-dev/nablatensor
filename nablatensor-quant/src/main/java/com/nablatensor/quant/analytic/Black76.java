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
package com.nablatensor.quant.analytic;

import com.nablatensor.quant.OptionType;

/**
 * Black's model (1976) — a European option on a forward or futures price
 * {@code F}, discounted at {@code r}. The cost of carry is zero, so
 * {@code d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt(T))} and the price is
 * {@code e^{-rT} [F N(d1) - K N(d2)]} for a call.
 *
 * <p>This is the standard-market-model pricer for caps and floors (each caplet a
 * call on the forward rate), European swaptions (a call/put on the forward swap
 * rate times the annuity), and options on bond and commodity futures.
 *
 * <p>In {@link AnalyticGreeks}: {@link AnalyticGreeks#delta()} is {@code dV/dF},
 * {@link AnalyticGreeks#rho()} is {@code dV/dr = -T * price} (the forward is held
 * fixed as {@code r} moves — the usual Black-76 convention).
 */
public final class Black76 {

  private Black76() {
  }

  /** Undiscounted-forward price: {@code r = 0}, so the discount factor is 1. */
  public static AnalyticGreeks of(OptionType type, double forward, double strike, double maturity, double vol) {
    return of(type, forward, strike, maturity, 0.0, vol);
  }

  /**
   * @param forward  forward / futures price {@code F}
   * @param strike   strike {@code K}
   * @param maturity time to expiry in years {@code T}
   * @param rate     continuously-compounded discount rate {@code r}
   * @param vol      lognormal volatility of the forward {@code sigma}
   */
  public static AnalyticGreeks of(OptionType type, double forward, double strike, double maturity,
                                  double rate, double vol) {
    return CostOfCarry.greeksCarryFixed(type, forward, strike, maturity, rate, 0.0, vol);
  }

  /** Bare price without the {@link AnalyticGreeks} wrapper. */
  public static double price(OptionType type, double forward, double strike, double maturity,
                             double rate, double vol) {
    return CostOfCarry.price(type, forward, strike, maturity, rate, 0.0, vol);
  }
}
