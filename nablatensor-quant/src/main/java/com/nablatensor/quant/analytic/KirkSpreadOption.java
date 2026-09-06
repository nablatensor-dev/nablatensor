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
 * Kirk's (1995) approximation for a European spread option — payoff
 * {@code max(S1_T - S2_T - K, 0)} — the standard closed form for a spark or dark
 * spread. It treats the spread as a Margrabe exchange between {@code F1} and
 * {@code F2 + K}, with an effective volatility that blends {@code vol1},
 * {@code vol2} and the moneyness of the second forward:
 *
 * <pre>{@code
 * F1, F2  = S1 e^{(r - q1) T},  S2 e^{(r - q2) T}
 * a       = F2 / (F2 + K)
 * sigma   = sqrt( vol1^2 - 2 rho vol1 vol2 a + vol2^2 a^2 )
 * d1      = (ln(F1 / (F2 + K)) + sigma^2 T / 2) / (sigma sqrt(T))
 * price   = e^{-rT} [ F1 N(d1) - (F2 + K) N(d1 - sigma sqrt(T)) ]
 * }</pre>
 *
 * <p>As {@code K -> 0} it collapses to the exact {@link Margrabe} price.
 */
public final class KirkSpreadOption {

  private KirkSpreadOption() {
  }

  public static double price(double s1, double s2, double strike, double vol1, double vol2, double rho,
                             double rate, double yield1, double yield2, double maturity) {
    double f1 = s1 * Math.exp((rate - yield1) * maturity);
    double f2 = s2 * Math.exp((rate - yield2) * maturity);
    double disc = Math.exp(-rate * maturity);

    if (maturity <= 0.0) {
      return Math.max(s1 - s2 - strike, 0.0);
    }
    double denom = f2 + strike;
    double a = f2 / denom;
    double sigma = Math.sqrt(Math.max(vol1 * vol1 - 2.0 * rho * vol1 * vol2 * a + vol2 * vol2 * a * a, 0.0));
    if (sigma <= 0.0) {
      return disc * Math.max(f1 - denom, 0.0);
    }
    double sqrtT = Math.sqrt(maturity);
    double d1 = (Math.log(f1 / denom) + 0.5 * sigma * sigma * maturity) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    return disc * (f1 * Normal.cdf(d1) - denom * Normal.cdf(d2));
  }

  /** Price and the two spot deltas by central differencing the closed form. */
  public static AnalyticGreeks greeks(double s1, double s2, double strike, double vol1, double vol2,
                                      double rho, double rate, double yield1, double yield2, double maturity) {
    double h1 = 1e-5 * Math.max(1.0, s1);
    double h2 = 1e-5 * Math.max(1.0, s2);
    double px = price(s1, s2, strike, vol1, vol2, rho, rate, yield1, yield2, maturity);
    double d1 = (price(s1 + h1, s2, strike, vol1, vol2, rho, rate, yield1, yield2, maturity)
        - price(s1 - h1, s2, strike, vol1, vol2, rho, rate, yield1, yield2, maturity)) / (2 * h1);
    double d2 = (price(s1, s2 + h2, strike, vol1, vol2, rho, rate, yield1, yield2, maturity)
        - price(s1, s2 - h2, strike, vol1, vol2, rho, rate, yield1, yield2, maturity)) / (2 * h2);
    double g1 = (price(s1 + h1, s2, strike, vol1, vol2, rho, rate, yield1, yield2, maturity)
        - 2 * px
        + price(s1 - h1, s2, strike, vol1, vol2, rho, rate, yield1, yield2, maturity)) / (h1 * h1);
    // delta2 packed into strikeSensitivity, gamma is d^2/dS1^2; the rest left zero.
    return new AnalyticGreeks(px, d1, g1, 0.0, 0.0, 0.0, d2);
  }

  /** Convenience alias for a put-style spread: {@code max(K - (S1 - S2), 0)} by parity. */
  public static double price(OptionType type, double s1, double s2, double strike, double vol1, double vol2,
                             double rho, double rate, double yield1, double yield2, double maturity) {
    double call = price(s1, s2, strike, vol1, vol2, rho, rate, yield1, yield2, maturity);
    if (type == OptionType.CALL) {
      return call;
    }
    double f1 = s1 * Math.exp((rate - yield1) * maturity);
    double f2 = s2 * Math.exp((rate - yield2) * maturity);
    return call - Math.exp(-rate * maturity) * (f1 - f2 - strike);
  }
}
