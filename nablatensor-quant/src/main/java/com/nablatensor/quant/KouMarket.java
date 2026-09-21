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
 * A Kou (2002) double-exponential jump-diffusion market: like {@link MertonJumpMarket}
 * but the log jump size is asymmetric two-sided exponential — up-jumps are
 * {@code Exp(etaUp)} with probability {@code probUp}, down-jumps are
 * {@code -Exp(etaDown)} otherwise. This reproduces the sharp peak and heavy,
 * asymmetric tails of equity returns better than a lognormal jump.
 *
 * <p>{@code etaUp > 1} is required for the compensator to be finite.
 *
 * @param spot          spot {@code S0}
 * @param strike        strike {@code K}
 * @param vol           diffusion volatility {@code sigma}
 * @param rate          continuously-compounded risk-free rate {@code r}
 * @param maturity      time to expiry in years {@code T}
 * @param jumpIntensity Poisson intensity {@code lambda}
 * @param probUp        probability a jump is upward
 * @param etaUp         rate of the upward exponential ({@code > 1})
 * @param etaDown       rate of the downward exponential
 */
@Of
public final class KouMarket {

  private KouMarket(double spot, double strike, double vol, double rate, double maturity, double jumpIntensity, double probUp, double etaUp, double etaDown) {
    this.spot = spot;
    this.strike = strike;
    this.vol = vol;
    this.rate = rate;
    this.maturity = maturity;
    this.jumpIntensity = jumpIntensity;
    this.probUp = probUp;
    this.etaUp = etaUp;
    this.etaDown = etaDown;
  }

  private final double spot;
  private final double strike;
  private final double vol;
  private final double rate;
  private final double maturity;
  private final double jumpIntensity;
  private final double probUp;
  private final double etaUp;
  private final double etaDown;
  static KouMarket create(double spot, double strike, double vol, double rate, double maturity, double jumpIntensity, double probUp, double etaUp, double etaDown) {
    return new KouMarket(spot, strike, vol, rate, maturity, jumpIntensity, probUp, etaUp, etaDown);
  }

  public static KouMarketBuilder of() {
    return new KouMarketBuilder();
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

  public double probUp() {
    return probUp;
  }

  public double etaUp() {
    return etaUp;
  }

  public double etaDown() {
    return etaDown;
  }


  public KouMarket validated() {
    if (!(spot > 0 && strike > 0 && vol >= 0 && maturity >= 0 && jumpIntensity >= 0
        && probUp >= 0 && probUp <= 1 && etaUp > 1.0 && etaDown > 0)) {
      throw new IllegalArgumentException("invalid Kou market: " + this);
    }
    return this;
  }

  public static KouMarket base() {
    return KouMarket.of().spot(100.0).strike(100.0).vol(0.16).rate(0.03).maturity(1.0).jumpIntensity(1.0).probUp(0.4).etaUp(10.0).etaDown(5.0).build();
  }
}
