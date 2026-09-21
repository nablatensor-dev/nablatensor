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

/**
 * Margrabe's (1978) closed form for a European option to exchange asset 2 for
 * asset 1 — payoff {@code max(S1_T - S2_T, 0)}. The option is on the ratio, so
 * only the effective volatility {@code sqrt(vol1^2 + vol2^2 - 2 rho vol1 vol2)}
 * and the two carry rates enter; there is no strike and no separate discount
 * term for the spread itself.
 *
 * <pre>{@code
 * sigma = sqrt(vol1^2 + vol2^2 - 2 rho vol1 vol2)
 * d1 = (ln(S1/S2) + (q2 - q1 + sigma^2/2) T) / (sigma sqrt(T))
 * d2 = d1 - sigma sqrt(T)
 * price = S1 e^{-q1 T} N(d1) - S2 e^{-q2 T} N(d2)
 * }</pre>
 */
public final class Margrabe {

  private final double price, delta1, delta2;
  private Margrabe(double price, double delta1, double delta2) { this.price=price; this.delta1=delta1; this.delta2=delta2; }
  public double price() { return price; }
  public double delta1() { return delta1; }
  public double delta2() { return delta2; }

  /**
   * @param s1        spot of the asset received
   * @param s2        spot of the asset given up
   * @param vol1      lognormal vol of asset 1
   * @param vol2      lognormal vol of asset 2
   * @param rho       correlation of the two Brownians
   * @param yield1    carry / convenience yield on asset 1
   * @param yield2    carry / convenience yield on asset 2
   * @param maturity  time to expiry in years
   */
  private static Margrabe calculate(double s1, double s2, double vol1, double vol2, double rho,
                            double yield1, double yield2, double maturity) {
    double sigma = Math.sqrt(Math.max(vol1 * vol1 + vol2 * vol2 - 2.0 * rho * vol1 * vol2, 0.0));
    if (maturity <= 0.0 || sigma <= 0.0) {
      double intrinsic = Math.max(s1 * Math.exp(-yield1 * maturity) - s2 * Math.exp(-yield2 * maturity), 0.0);
      return new Margrabe(intrinsic, 0.0, 0.0);
    }
    double sqrtT = Math.sqrt(maturity);
    double d1 = (Math.log(s1 / s2) + (yield2 - yield1 + 0.5 * sigma * sigma) * maturity) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    double disc1 = Math.exp(-yield1 * maturity);
    double disc2 = Math.exp(-yield2 * maturity);
    double price = s1 * disc1 * Normal.cdf(d1) - s2 * disc2 * Normal.cdf(d2);
    return new Margrabe(price, disc1 * Normal.cdf(d1), -disc2 * Normal.cdf(d2));
  }

  public static Builder of() { return new Builder(); }

  public static final class Builder {
    private Double s1, s2, vol1, vol2, rho, yield1, yield2, maturity;
    private Builder() {}
    public Builder s1(double v) { s1=v; return this; }
    public Builder s2(double v) { s2=v; return this; }
    public Builder vol1(double v) { vol1=v; return this; }
    public Builder vol2(double v) { vol2=v; return this; }
    public Builder rho(double v) { rho=v; return this; }
    public Builder yield1(double v) { yield1=v; return this; }
    public Builder yield2(double v) { yield2=v; return this; }
    public Builder maturity(double v) { maturity=v; return this; }
    public Margrabe build() { return calculate(req(s1,"s1"),req(s2,"s2"),req(vol1,"vol1"),req(vol2,"vol2"),
        req(rho,"rho"),req(yield1,"yield1"),req(yield2,"yield2"),req(maturity,"maturity")); }
  }
  private static <T> T req(T v, String n) { if (v == null) throw new IllegalStateException("Required field " + n + " is not set"); return v; }

}