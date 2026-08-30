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

/**
 * Closed-form Black-Scholes price and Greeks for a European option, used as the
 * analytic reference the Monte-Carlo adjoint result is checked against.
 *
 * <p>Sign conventions match {@link Pricing}: {@link #delta} is dV/dS0,
 * {@link #vega} is dV/dsigma (per unit vol, not per 1%), {@link #rho} is dV/dr,
 * {@link #strikeSensitivity} is dV/dK.
 */
public record BlackScholes(double price, double delta, double vega, double rho,
                           double strikeSensitivity) {

  public static BlackScholes of(OptionType type, EquityMarket m) {
    double s = m.spot();
    double k = m.strike();
    double t = m.maturity();
    double r = m.rate();
    double sigma = m.vol();
    double disc = Math.exp(-r * t);

    if (t == 0.0 || sigma == 0.0) {
      double intrinsic = Math.max(type.sign() * (s - k * disc), 0.0);
      return new BlackScholes(intrinsic, 0.0, 0.0, 0.0, 0.0);
    }

    double sqrtT = Math.sqrt(t);
    double d1 = (Math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    double pdf = phi(d1);

    if (type == OptionType.CALL) {
      double nd1 = N(d1);
      double nd2 = N(d2);
      return new BlackScholes(
          s * nd1 - k * disc * nd2,
          nd1,
          s * pdf * sqrtT,
          k * t * disc * nd2,
          -disc * nd2);
    } else {
      double nnd1 = N(-d1);
      double nnd2 = N(-d2);
      return new BlackScholes(
          k * disc * nnd2 - s * nnd1,
          -nnd1,
          s * pdf * sqrtT,
          -k * t * disc * nnd2,
          disc * nnd2);
    }
  }

  /** Standard normal PDF. */
  public static double phi(double x) {
    return Math.exp(-0.5 * x * x) / Math.sqrt(2.0 * Math.PI);
  }

  /** Standard normal CDF via a rational erfc approximation (abs error &lt; 1.5e-7). */
  public static double N(double x) {
    return 0.5 * erfc(-x / Math.sqrt(2.0));
  }

  private static double erfc(double x) {
    double z = Math.abs(x);
    double s = 1.0 / (1.0 + 0.5 * z);
    double ans = s * Math.exp(-z * z - 1.26551223
        + s * (1.00002368
        + s * (0.37409196
        + s * (0.09678418
        + s * (-0.18628806
        + s * (0.27886807
        + s * (-1.13520398
        + s * (1.48851587
        + s * (-0.82215223
        + s * 0.17087277)))))))));
    return x >= 0.0 ? ans : 2.0 - ans;
  }
}
