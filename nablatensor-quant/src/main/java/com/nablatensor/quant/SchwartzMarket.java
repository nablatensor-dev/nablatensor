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
 * A Schwartz (1997) one-factor commodity market: the log spot mean-reverts,
 *
 * <pre>{@code
 * d ln S = kappa (level - ln S) dt + sigma dW
 * }</pre>
 *
 * where {@code level} is the <em>risk-neutral</em> long-run log price (the
 * market price of risk has been folded in). Every field is a differentiable
 * input.
 *
 * @param spot     current spot {@code S0}
 * @param kappa    mean-reversion speed
 * @param level    risk-neutral long-run log price {@code alpha}
 * @param sigma    volatility of the log spot
 * @param rate     flat discount rate
 */
@Of
public final class SchwartzMarket {

  private SchwartzMarket(double spot, double kappa, double level, double sigma, double rate) {
    this.spot = spot;
    this.kappa = kappa;
    this.level = level;
    this.sigma = sigma;
    this.rate = rate;
  }

  private final double spot;
  private final double kappa;
  private final double level;
  private final double sigma;
  private final double rate;
  static SchwartzMarket create(double spot, double kappa, double level, double sigma, double rate) {
    return new SchwartzMarket(spot, kappa, level, sigma, rate);
  }

  public static SchwartzMarketBuilder of() {
    return new SchwartzMarketBuilder();
  }

  public double spot() {
    return spot;
  }

  public double kappa() {
    return kappa;
  }

  public double level() {
    return level;
  }

  public double sigma() {
    return sigma;
  }

  public double rate() {
    return rate;
  }


  public SchwartzMarket validated() {
    if (!(spot > 0 && kappa > 0 && sigma >= 0)) {
      throw new IllegalArgumentException("invalid Schwartz market: " + this);
    }
    return this;
  }

  public static SchwartzMarket base() {
    return SchwartzMarket.of().spot(50.0).kappa(1.2).level(Math.log(55.0)).sigma(0.30).rate(0.03).build();
  }
}
