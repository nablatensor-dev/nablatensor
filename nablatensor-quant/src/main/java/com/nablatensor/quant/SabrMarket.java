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
public record SabrMarket(double forward, double strike, double rate,
                         double alpha, double beta, double rho, double nu) {

  public SabrMarket validated() {
    if (!(forward > 0 && strike > 0 && alpha > 0 && beta >= 0 && beta <= 1
        && rho > -1 && rho < 1 && nu >= 0)) {
      throw new IllegalArgumentException("invalid SABR market: " + this);
    }
    return this;
  }

  public static SabrMarket atm() {
    return new SabrMarket(0.05, 0.05, 0.0, 0.20, 0.5, -0.3, 0.4);
  }

  public SabrMarket withStrike(double k) {
    return new SabrMarket(forward, k, rate, alpha, beta, rho, nu);
  }

  public SabrMarket with(double alpha, double beta, double rho, double nu) {
    return new SabrMarket(forward, strike, rate, alpha, beta, rho, nu);
  }
}
