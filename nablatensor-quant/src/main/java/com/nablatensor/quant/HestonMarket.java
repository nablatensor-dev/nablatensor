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
public record HestonMarket(double spot, double strike, double rate,
                           double v0, double kappa, double theta, double xi, double rho) {

  public HestonMarket validated() {
    if (!(spot > 0 && strike > 0 && v0 >= 0 && kappa >= 0 && theta >= 0 && xi >= 0
        && rho > -1.0 && rho < 1.0)) {
      throw new IllegalArgumentException("invalid Heston market: " + this);
    }
    return this;
  }

  /** A commonly-cited test parameter set (Andersen 2008), r=0. */
  public static HestonMarket andersenCase1() {
    return new HestonMarket(100.0, 100.0, 0.0, 0.04, 0.5, 0.04, 1.0, -0.9);
  }

  public HestonMarket withSpot(double s) {
    return new HestonMarket(s, strike, rate, v0, kappa, theta, xi, rho);
  }

  public HestonMarket withV0(double v) {
    return new HestonMarket(spot, strike, rate, v, kappa, theta, xi, rho);
  }
}
