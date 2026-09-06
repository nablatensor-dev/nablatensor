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
 * The generalised Black-Scholes-Merton price for a European option under a
 * lognormal underlying with a continuous cost of carry {@code b}:
 *
 * <pre>{@code
 * d1 = (ln(S/K) + (b + sigma^2/2) T) / (sigma sqrt(T))
 * d2 = d1 - sigma sqrt(T)
 * call = S e^{(b-r)T} N(d1) - K e^{-rT} N(d2)
 * put  = K e^{-rT} N(-d2) - S e^{(b-r)T} N(-d1)
 * }</pre>
 *
 * <p>The single carry parameter recovers the whole vanilla family:
 * <ul>
 *   <li>{@code b = r} — Black-Scholes on a non-dividend stock;</li>
 *   <li>{@code b = r - q} — a stock or index with continuous dividend yield
 *       {@code q} ({@link GeneralizedBsm});</li>
 *   <li>{@code b = r - r_f} — an FX rate with foreign rate {@code r_f}
 *       ({@link GarmanKohlhagen});</li>
 *   <li>{@code b = 0} — an option on a forward or futures price
 *       ({@link Black76}).</li>
 * </ul>
 *
 * <p>This class exposes the raw {@link #price} function; the public wrappers add
 * the {@link AnalyticGreeks} for their own parameterisation (which fixes how the
 * carry moves with {@code r}).
 */
public final class CostOfCarry {

  private CostOfCarry() {
  }

  /**
   * @param type  call or put
   * @param s     underlying level {@code S}
   * @param k     strike {@code K}
   * @param t     time to expiry in years {@code T}
   * @param r     continuously-compounded discount rate {@code r}
   * @param b     cost of carry {@code b}
   * @param sigma lognormal volatility {@code sigma}
   */
  public static double price(OptionType type, double s, double k, double t, double r, double b, double sigma) {
    if (t <= 0.0 || sigma <= 0.0) {
      double fwd = s * Math.exp(b * t);
      return Math.max(type.sign() * (fwd - k), 0.0) * Math.exp(-r * t);
    }
    double sqrtT = Math.sqrt(t);
    double d1 = (Math.log(s / k) + (b + 0.5 * sigma * sigma) * t) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    double carryDisc = Math.exp((b - r) * t);
    double disc = Math.exp(-r * t);
    if (type == OptionType.CALL) {
      return s * carryDisc * Normal.cdf(d1) - k * disc * Normal.cdf(d2);
    }
    return k * disc * Normal.cdf(-d2) - s * carryDisc * Normal.cdf(-d1);
  }

  /**
   * Price and Greeks with the carry held <em>independent</em> of {@code r} — so
   * {@link AnalyticGreeks#rho()} is the pure discounting term {@code -T * price}.
   * {@link Black76} uses this directly; the dividend-yield and FX wrappers add
   * the carry's own {@code r}-dependence on top.
   */
  static AnalyticGreeks greeksCarryFixed(OptionType type, double s, double k, double t,
                                         double r, double b, double sigma) {
    if (t <= 0.0 || sigma <= 0.0) {
      return AnalyticGreeks.intrinsic(price(type, s, k, t, r, b, sigma));
    }
    return Greeking.central((ss, kk, tt, rr, vv) -> price(type, ss, kk, tt, rr, b, vv), s, k, t, r, sigma);
  }

  /**
   * The carry sensitivity {@code dV/db} at fixed {@code r}, in closed form:
   * {@code +T S e^{(b-r)T} N(d1)} for a call, {@code -T S e^{(b-r)T} N(-d1)} for
   * a put. The dividend-yield and FX wrappers chain this through {@code db/dr}
   * and {@code db/dq} (or {@code db/dr_f}).
   */
  static double carryRho(OptionType type, double s, double k, double t, double r, double b, double sigma) {
    if (t <= 0.0 || sigma <= 0.0) {
      return 0.0;
    }
    double sqrtT = Math.sqrt(t);
    double d1 = (Math.log(s / k) + (b + 0.5 * sigma * sigma) * t) / (sigma * sqrtT);
    double carryDisc = Math.exp((b - r) * t);
    return type == OptionType.CALL
        ? t * s * carryDisc * Normal.cdf(d1)
        : -t * s * carryDisc * Normal.cdf(-d1);
  }
}
