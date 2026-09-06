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
 * The Bachelier (normal) model — the forward is arithmetic Brownian rather than
 * geometric, so the price can be negative and the volatility {@code sigmaN} is
 * quoted in price units per {@code sqrt(year)} rather than as a fraction. This
 * is the market model for interest-rate options in a low- or negative-rate
 * regime, where a lognormal forward is inadmissible.
 *
 * <pre>{@code
 * d    = (F - K) / (sigmaN sqrt(T))
 * call = e^{-rT} [ (F - K) N(d) + sigmaN sqrt(T) phi(d) ]
 * put  = e^{-rT} [ (K - F) N(-d) + sigmaN sqrt(T) phi(d) ]
 * }</pre>
 *
 * <p>In {@link AnalyticGreeks}: {@link AnalyticGreeks#delta()} is {@code dV/dF},
 * {@link AnalyticGreeks#vega()} is {@code dV/dsigmaN} (per unit of normal vol).
 */
public final class Bachelier {

  private Bachelier() {
  }

  public static AnalyticGreeks of(OptionType type, double forward, double strike, double maturity,
                                  double normalVol) {
    return of(type, forward, strike, maturity, 0.0, normalVol);
  }

  /**
   * @param forward   forward price / rate {@code F}
   * @param strike    strike {@code K}
   * @param maturity  time to expiry in years {@code T}
   * @param rate      continuously-compounded discount rate {@code r}
   * @param normalVol absolute (normal) volatility {@code sigmaN}, price units per sqrt(year)
   */
  public static AnalyticGreeks of(OptionType type, double forward, double strike, double maturity,
                                  double rate, double normalVol) {
    if (maturity <= 0.0 || normalVol <= 0.0) {
      return AnalyticGreeks.intrinsic(price(type, forward, strike, maturity, rate, normalVol));
    }
    return Greeking.central((f, k, t, r, v) -> price(type, f, k, t, r, v),
        forward, strike, maturity, rate, normalVol);
  }

  /** Bare price. */
  public static double price(OptionType type, double forward, double strike, double maturity,
                             double rate, double normalVol) {
    double disc = Math.exp(-rate * maturity);
    if (maturity <= 0.0 || normalVol <= 0.0) {
      return Math.max(type.sign() * (forward - strike), 0.0) * disc;
    }
    double stdev = normalVol * Math.sqrt(maturity);
    double d = (forward - strike) / stdev;
    if (type == OptionType.CALL) {
      return disc * ((forward - strike) * Normal.cdf(d) + stdev * Normal.pdf(d));
    }
    return disc * ((strike - forward) * Normal.cdf(-d) + stdev * Normal.pdf(d));
  }
}
