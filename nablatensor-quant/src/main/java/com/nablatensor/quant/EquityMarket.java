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
public record EquityMarket(double spot, double strike, double vol, double rate, double maturity) {

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
    return new EquityMarket(100.0, 100.0, 0.20, 0.03, 1.0);
  }

  public EquityMarket withSpot(double spot) {
    return new EquityMarket(spot, strike, vol, rate, maturity);
  }

  public EquityMarket withStrike(double strike) {
    return new EquityMarket(spot, strike, vol, rate, maturity);
  }

  public EquityMarket withVol(double vol) {
    return new EquityMarket(spot, strike, vol, rate, maturity);
  }

  public EquityMarket withRate(double rate) {
    return new EquityMarket(spot, strike, vol, rate, maturity);
  }

  public EquityMarket withMaturity(double maturity) {
    return new EquityMarket(spot, strike, vol, rate, maturity);
  }
}
