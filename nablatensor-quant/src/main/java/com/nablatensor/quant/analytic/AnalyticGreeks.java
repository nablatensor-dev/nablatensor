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

import com.nablatensor.codegen.Of;

/**
 * A closed-form price and its first- and second-order sensitivities — the
 * reference an adjoint Monte-Carlo run is checked against.
 *
 * <p>Sign conventions match {@link com.nablatensor.quant.BlackScholes} and the
 * gradient carrier {@link com.nablatensor.quant.EquityMarket}:
 * <ul>
 *   <li>{@link #delta} is {@code dV/dS} (spot delta of the primary underlying);</li>
 *   <li>{@link #gamma} is {@code d^2V/dS^2};</li>
 *   <li>{@link #vega} is {@code dV/dsigma}, per unit volatility (not per 1%);</li>
 *   <li>{@link #theta} is {@code dV/dT}, the derivative with respect to the
 *       <em>time to expiry</em> — the same convention as
 *       {@code EquityMarket.maturity()} as a gradient slot, so it is the negative
 *       of the calendar-time theta usually quoted in textbooks;</li>
 *   <li>{@link #rho} is the total {@code dV/dr} in the pricer's own
 *       parameterisation (for a dividend-yield or foreign-rate model the carry
 *       moves with {@code r});</li>
 *   <li>{@link #strikeSensitivity} is {@code dV/dK}.</li>
 * </ul>
 *
 * <p>Every pricer in this package computes {@link #price} in closed form and the
 * six sensitivities by central differencing of that closed form (see
 * {@link Greeking}); the differences carry no Monte-Carlo noise and only
 * {@code ~1e-7} truncation error, which is what makes the record usable as an
 * oracle.
 */
@Of
public final class AnalyticGreeks {

  private final double price;
  private final double delta;
  private final double gamma;
  private final double vega;
  private final double theta;
  private final double rho;
  private final double strikeSensitivity;

  private AnalyticGreeks(double price, double delta, double gamma, double vega, double theta, double rho, double strikeSensitivity) {
    this.price = price;
    this.delta = delta;
    this.gamma = gamma;
    this.vega = vega;
    this.theta = theta;
    this.rho = rho;
    this.strikeSensitivity = strikeSensitivity;
  }

  static AnalyticGreeks create(double price, double delta, double gamma, double vega, double theta, double rho, double strikeSensitivity) {
    return new AnalyticGreeks(price, delta, gamma, vega, theta, rho, strikeSensitivity);
  }

  public static AnalyticGreeksBuilder of() {
    return new AnalyticGreeksBuilder();
  }

  public double price() {
    return price;
  }

  public double delta() {
    return delta;
  }

  public double gamma() {
    return gamma;
  }

  public double vega() {
    return vega;
  }

  public double theta() {
    return theta;
  }

  public double rho() {
    return rho;
  }

  public double strikeSensitivity() {
    return strikeSensitivity;
  }


  /** A price with all sensitivities zero — the degenerate {@code T=0} or {@code sigma=0} case. */
  public static AnalyticGreeks intrinsic(double price) {
    return AnalyticGreeks.of().price(price).delta(0.0).gamma(0.0).vega(0.0).theta(0.0).rho(0.0).strikeSensitivity(0.0).build();
  }
}
