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
 * A Hull-White one-factor market.
 *
 * @param r0            initial short rate
 * @param level         flat long-run level {@code b} the rate reverts to
 * @param meanReversion reversion speed {@code a}
 * @param sigma         absolute (normal) short-rate volatility
 * @param strike        strike for a rate option, e.g. a caplet
 */
@Of
public final class HullWhiteMarket {

  private HullWhiteMarket(double r0, double level, double meanReversion, double sigma, double strike) {
    this.r0 = r0;
    this.level = level;
    this.meanReversion = meanReversion;
    this.sigma = sigma;
    this.strike = strike;
  }

  private final double r0;
  private final double level;
  private final double meanReversion;
  private final double sigma;
  private final double strike;
  static HullWhiteMarket create(double r0, double level, double meanReversion, double sigma, double strike) {
    return new HullWhiteMarket(r0, level, meanReversion, sigma, strike);
  }

  public static HullWhiteMarketBuilder of() {
    return new HullWhiteMarketBuilder();
  }

  public double r0() {
    return r0;
  }

  public double level() {
    return level;
  }

  public double meanReversion() {
    return meanReversion;
  }

  public double sigma() {
    return sigma;
  }

  public double strike() {
    return strike;
  }


  public HullWhiteMarket validated() {
    if (!(meanReversion > 0 && sigma >= 0)) {
      throw new IllegalArgumentException("invalid Hull-White market: " + this);
    }
    return this;
  }

  public static HullWhiteMarket base() {
    return HullWhiteMarket.of().r0(0.03).level(0.03).meanReversion(0.10).sigma(0.01).strike(0.03).build();
  }
}
