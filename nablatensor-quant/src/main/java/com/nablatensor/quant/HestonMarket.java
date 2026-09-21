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
 * A Heston market: the vanilla spot / strike / rate plus the five model
 * parameters. Every field is a differentiable input; a run hands back an
 * {@code HestonMarket} of the same shape carrying the sensitivity to each.
 *
 * @param spot   S0
 * @param strike K
 * @param rate   flat continuously-compounded rate r
 * @param v0     initial instantaneous variance
 * @param kappa  mean-reversion speed of the variance
 * @param theta  long-run variance
 * @param xi     vol-of-vol
 * @param rho    spot/variance Brownian correlation, in (-1, 1)
 */
@Of
public final class HestonMarket {

  private HestonMarket(double spot, double strike, double rate, double v0, double kappa, double theta, double xi, double rho) {
    this.spot = spot;
    this.strike = strike;
    this.rate = rate;
    this.v0 = v0;
    this.kappa = kappa;
    this.theta = theta;
    this.xi = xi;
    this.rho = rho;
  }

  private final double spot;
  private final double strike;
  private final double rate;
  private final double v0;
  private final double kappa;
  private final double theta;
  private final double xi;
  private final double rho;
  static HestonMarket create(double spot, double strike, double rate, double v0, double kappa, double theta, double xi, double rho) {
    return new HestonMarket(spot, strike, rate, v0, kappa, theta, xi, rho);
  }

  public static HestonMarketBuilder of() {
    return new HestonMarketBuilder();
  }

  public double spot() {
    return spot;
  }

  public double strike() {
    return strike;
  }

  public double rate() {
    return rate;
  }

  public double v0() {
    return v0;
  }

  public double kappa() {
    return kappa;
  }

  public double theta() {
    return theta;
  }

  public double xi() {
    return xi;
  }

  public double rho() {
    return rho;
  }


  public HestonMarket validated() {
    if (!(spot > 0 && strike > 0 && v0 >= 0 && kappa >= 0 && theta >= 0 && xi >= 0
        && rho > -1.0 && rho < 1.0)) {
      throw new IllegalArgumentException("invalid Heston market: " + this);
    }
    return this;
  }

  /** A commonly-cited test parameter set (Andersen 2008), r=0. */
  public static HestonMarket andersenCase1() {
    return HestonMarket.of().spot(100.0).strike(100.0).rate(0.0).v0(0.04).kappa(0.5).theta(0.04).xi(1.0).rho(-0.9).build();
  }

  public HestonMarket withSpot(double s) {
    return HestonMarket.of().spot(s).strike(strike).rate(rate).v0(v0).kappa(kappa).theta(theta).xi(xi).rho(rho).build();
  }

  public HestonMarket withV0(double v) {
    return HestonMarket.of().spot(spot).strike(strike).rate(rate).v0(v).kappa(kappa).theta(theta).xi(xi).rho(rho).build();
  }
}
