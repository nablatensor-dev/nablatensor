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

import com.nablatensor.codegen.Of;

/**
 * One equity underlying and the flat-rate world it lives in.
 *
 * <p>Every component is a differentiable input to a recorded valuation, so the
 * gradient handed back by a Monte-Carlo run is an {@code EquityMarket} of the
 * same shape: {@link #spot()} carries delta, {@link #vol()} vega, {@link #rate()}
 * rho, {@link #strike()} the strike sensitivity and {@link #maturity()} the
 * sensitivity to time to expiry.
 *
 * @param spot     current price of the underlying, {@code S0}
 * @param strike   option strike, {@code K}
 * @param vol      lognormal volatility, {@code sigma} (annualised)
 * @param rate     continuously-compounded risk-free rate, {@code r}
 * @param maturity time to expiry in years, {@code T}
 */
@Of
public final class EquityMarket {

  private EquityMarket(double spot, double strike, double vol, double rate, double maturity) {
    this.spot = spot;
    this.strike = strike;
    this.vol = vol;
    this.rate = rate;
    this.maturity = maturity;
  }

  private final double spot;
  private final double strike;
  private final double vol;
  private final double rate;
  private final double maturity;
  static EquityMarket create(double spot, double strike, double vol, double rate, double maturity) {
    return new EquityMarket(spot, strike, vol, rate, maturity);
  }

  public static EquityMarketBuilder of() {
    return new EquityMarketBuilder();
  }

  public double spot() {
    return spot;
  }

  public double strike() {
    return strike;
  }

  public double vol() {
    return vol;
  }

  public double rate() {
    return rate;
  }

  public double maturity() {
    return maturity;
  }


  /**
   * Rejects a market that a payoff cannot be simulated against. Not enforced in
   * the constructor: the engine also reuses this record's shape as the carrier
   * for the gradient vector, whose components carry no such constraints.
   */
  public EquityMarket validated() {
    if (!(vol >= 0.0) || !(maturity >= 0.0) || !(spot > 0.0) || !(strike > 0.0)) {
      throw new IllegalArgumentException(
          "need spot>0, strike>0, vol>=0, maturity>=0; got "
              + spot + "/" + strike + "/" + vol + "/" + maturity);
    }
    return this;
  }

  /** A textbook at-the-money one-year call market: S0=K=100, sigma=20%, r=3%. */
  public static EquityMarket atmOneYear() {
    return EquityMarket.of().spot(100.0).strike(100.0).vol(0.20).rate(0.03).maturity(1.0).build();
  }

  public EquityMarket withSpot(double spot) {
    return EquityMarket.of().spot(spot).strike(strike).vol(vol).rate(rate).maturity(maturity).build();
  }

  public EquityMarket withStrike(double strike) {
    return EquityMarket.of().spot(spot).strike(strike).vol(vol).rate(rate).maturity(maturity).build();
  }

  public EquityMarket withVol(double vol) {
    return EquityMarket.of().spot(spot).strike(strike).vol(vol).rate(rate).maturity(maturity).build();
  }

  public EquityMarket withRate(double rate) {
    return EquityMarket.of().spot(spot).strike(strike).vol(vol).rate(rate).maturity(maturity).build();
  }

  public EquityMarket withMaturity(double maturity) {
    return EquityMarket.of().spot(spot).strike(strike).vol(vol).rate(rate).maturity(maturity).build();
  }
}
