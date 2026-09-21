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
 * A SABR market: the forward, strike and discount rate plus the four model
 * parameters {@code alpha} (initial vol), {@code beta}, {@code rho}, {@code nu}.
 *
 * @param forward F0
 * @param strike  K
 * @param rate    flat discount rate
 * @param alpha   initial stochastic volatility
 * @param beta    CEV exponent, in [0, 1]
 * @param rho     forward/vol correlation, in (-1, 1)
 * @param nu      vol-of-vol
 */
@Of
public final class SabrMarket {

  private SabrMarket(double forward, double strike, double rate, double alpha, double beta, double rho, double nu) {
    this.forward = forward;
    this.strike = strike;
    this.rate = rate;
    this.alpha = alpha;
    this.beta = beta;
    this.rho = rho;
    this.nu = nu;
  }

  private final double forward;
  private final double strike;
  private final double rate;
  private final double alpha;
  private final double beta;
  private final double rho;
  private final double nu;
  static SabrMarket create(double forward, double strike, double rate, double alpha, double beta, double rho, double nu) {
    return new SabrMarket(forward, strike, rate, alpha, beta, rho, nu);
  }

  public static SabrMarketBuilder of() {
    return new SabrMarketBuilder();
  }

  public double forward() {
    return forward;
  }

  public double strike() {
    return strike;
  }

  public double rate() {
    return rate;
  }

  public double alpha() {
    return alpha;
  }

  public double beta() {
    return beta;
  }

  public double rho() {
    return rho;
  }

  public double nu() {
    return nu;
  }


  public SabrMarket validated() {
    if (!(forward > 0 && strike > 0 && alpha > 0 && beta >= 0 && beta <= 1
        && rho > -1 && rho < 1 && nu >= 0)) {
      throw new IllegalArgumentException("invalid SABR market: " + this);
    }
    return this;
  }

  public static SabrMarket atm() {
    return SabrMarket.of().forward(0.05).strike(0.05).rate(0.0).alpha(0.20).beta(0.5).rho(-0.3).nu(0.4).build();
  }

  public SabrMarket withStrike(double k) {
    return SabrMarket.of().forward(forward).strike(k).rate(rate).alpha(alpha).beta(beta).rho(rho).nu(nu).build();
  }

  public SabrMarket with(double alpha, double beta, double rho, double nu) {
    return SabrMarket.of().forward(forward).strike(strike).rate(rate).alpha(alpha).beta(beta).rho(rho).nu(nu).build();
  }
}
