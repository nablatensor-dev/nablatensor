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
 * A parametric local-volatility market. The reference spot the CEV ratio is
 * measured against is fixed at {@link #REF_SPOT}.
 *
 * @param spot   S0
 * @param strike K
 * @param rate   flat rate r
 * @param sigma0 volatility level at {@code S = REF_SPOT}
 * @param skew   CEV exponent on {@code S / REF_SPOT} (0 = GBM, negative = equity smile)
 */
@Of
public final class LocalVolMarket {

  private LocalVolMarket(double spot, double strike, double rate, double sigma0, double skew) {
    this.spot = spot;
    this.strike = strike;
    this.rate = rate;
    this.sigma0 = sigma0;
    this.skew = skew;
  }

  private final double spot;
  private final double strike;
  private final double rate;
  private final double sigma0;
  private final double skew;
  static LocalVolMarket create(double spot, double strike, double rate, double sigma0, double skew) {
    return new LocalVolMarket(spot, strike, rate, sigma0, skew);
  }

  public static LocalVolMarketBuilder of() {
    return new LocalVolMarketBuilder();
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

  public double sigma0() {
    return sigma0;
  }

  public double skew() {
    return skew;
  }


  /** The spot the local-vol ratio {@code (S / REF_SPOT)} is anchored to. */
  public static final double REF_SPOT = 100.0;

  public LocalVolMarket validated() {
    if (!(spot > 0 && strike > 0 && sigma0 > 0)) {
      throw new IllegalArgumentException("invalid local-vol market: " + this);
    }
    return this;
  }

  public static LocalVolMarket smile() {
    return LocalVolMarket.of().spot(100.0).strike(100.0).rate(0.02).sigma0(0.20).skew(-0.5).build();
  }
}
