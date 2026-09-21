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
 * A Merton (1976) jump-diffusion market: geometric Brownian motion of volatility
 * {@code vol} plus a compound-Poisson jump component of intensity
 * {@code jumpIntensity} whose multiplicative jump size is lognormal,
 * {@code ln Y ~ N(jumpMean, jumpVol^2)}.
 *
 * <p>Every field is a differentiable input; {@code jumpMean} and {@code jumpVol}
 * feed the jump-size distribution directly, while {@code jumpIntensity} enters
 * both the drift compensator and — via the smoothed per-step jump indicator —
 * the jump frequency.
 *
 * @param spot          spot {@code S0}
 * @param strike        strike {@code K}
 * @param vol           diffusion volatility {@code sigma}
 * @param rate          continuously-compounded risk-free rate {@code r}
 * @param maturity      time to expiry in years {@code T}
 * @param jumpIntensity Poisson intensity {@code lambda} (expected jumps per year)
 * @param jumpMean      mean of the log jump size {@code muJ}
 * @param jumpVol       standard deviation of the log jump size {@code deltaJ}
 */
@Of
public final class MertonJumpMarket {

  private MertonJumpMarket(double spot, double strike, double vol, double rate, double maturity, double jumpIntensity, double jumpMean, double jumpVol) {
    this.spot = spot;
    this.strike = strike;
    this.vol = vol;
    this.rate = rate;
    this.maturity = maturity;
    this.jumpIntensity = jumpIntensity;
    this.jumpMean = jumpMean;
    this.jumpVol = jumpVol;
  }

  private final double spot;
  private final double strike;
  private final double vol;
  private final double rate;
  private final double maturity;
  private final double jumpIntensity;
  private final double jumpMean;
  private final double jumpVol;
  static MertonJumpMarket create(double spot, double strike, double vol, double rate, double maturity, double jumpIntensity, double jumpMean, double jumpVol) {
    return new MertonJumpMarket(spot, strike, vol, rate, maturity, jumpIntensity, jumpMean, jumpVol);
  }

  public static MertonJumpMarketBuilder of() {
    return new MertonJumpMarketBuilder();
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

  public double jumpIntensity() {
    return jumpIntensity;
  }

  public double jumpMean() {
    return jumpMean;
  }

  public double jumpVol() {
    return jumpVol;
  }


  public MertonJumpMarket validated() {
    if (!(spot > 0 && strike > 0 && vol >= 0 && maturity >= 0 && jumpIntensity >= 0 && jumpVol >= 0)) {
      throw new IllegalArgumentException("invalid Merton jump market: " + this);
    }
    return this;
  }

  public static MertonJumpMarket base() {
    return MertonJumpMarket.of().spot(100.0).strike(100.0).vol(0.18).rate(0.03).maturity(1.0).jumpIntensity(0.75).jumpMean(-0.05).jumpVol(0.15).build();
  }
}
