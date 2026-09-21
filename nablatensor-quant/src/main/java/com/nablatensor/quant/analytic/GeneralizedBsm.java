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

import com.nablatensor.quant.OptionTypeEnum;

/**
 * Merton's extension of Black-Scholes to a continuous dividend yield {@code q} —
 * the pricer for options on a dividend-paying stock or a stock index. The cost
 * of carry is {@code b = r - q}.
 *
 * <p>{@link AnalyticGreeks#rho()} is the <em>total</em> {@code dV/dr} with
 * {@code q} held fixed (so the carry moves one-for-one with {@code r}); the
 * separate {@link #dividendRho} slot below is {@code dV/dq}.
 */
public final class GeneralizedBsm {

  private final AnalyticGreeks greeks;
  private final double dividendRho;

  private GeneralizedBsm(AnalyticGreeks greeks, double dividendRho) {
    this.greeks = greeks;
    this.dividendRho = dividendRho;
  }

  public AnalyticGreeks greeks() {
    return greeks;
  }

  public double dividendRho() {
    return dividendRho;
  }

  /**
   * @param type     call or put
   * @param spot     spot {@code S}
   * @param strike   strike {@code K}
   * @param maturity time to expiry in years {@code T}
   * @param rate     continuously-compounded risk-free rate {@code r}
   * @param dividend continuous dividend yield {@code q}
   * @param vol      lognormal volatility {@code sigma}
   */
  private static GeneralizedBsm calculate(OptionTypeEnum type, double spot, double strike, double maturity,
                                  double rate, double dividend, double vol) {
    double b = rate - dividend;
    // Price closed form; Greeks by differencing it with b = r - q tracking r.
    Greeking.Price5 f = (s, k, t, r, v) -> CostOfCarry.price(type, s, k, t, r, r - dividend, v);
    AnalyticGreeks g = (maturity <= 0.0 || vol <= 0.0)
        ? AnalyticGreeks.intrinsic(CostOfCarry.price(type, spot, strike, maturity, rate, b, vol))
        : Greeking.central(f, spot, strike, maturity, rate, vol);
    double dividendRho = -CostOfCarry.carryRho(type, spot, strike, maturity, rate, b, vol);
    return new GeneralizedBsm(g, dividendRho);
  }


  /** Starts a calculation with named Black-Scholes-Merton inputs. */
  public static Builder of() {
    return new Builder();
  }

  /** Named input builder for the generalized Black-Scholes-Merton result. */
  public static final class Builder {
    private OptionTypeEnum type;
    private Double spot;
    private Double strike;
    private Double maturity;
    private Double rate;
    private Double dividend;
    private Double vol;

    private Builder() {
    }

    public Builder type(OptionTypeEnum value) { type = value; return this; }
    public Builder spot(double value) { spot = value; return this; }
    public Builder strike(double value) { strike = value; return this; }
    public Builder maturity(double value) { maturity = value; return this; }
    public Builder rate(double value) { rate = value; return this; }
    public Builder dividend(double value) { dividend = value; return this; }
    public Builder vol(double value) { vol = value; return this; }

    public GeneralizedBsm build() {
      return calculate(required(type, "type"), required(spot, "spot"),
          required(strike, "strike"), required(maturity, "maturity"),
          required(rate, "rate"), required(dividend, "dividend"), required(vol, "vol"));
    }
  }

  private static <T> T required(T value, String name) {
    if (value == null) {
      throw new IllegalStateException("Required field '" + name + "' is not set");
    }
    return value;
  }

  public double price() {
    return greeks.price();
  }
}
