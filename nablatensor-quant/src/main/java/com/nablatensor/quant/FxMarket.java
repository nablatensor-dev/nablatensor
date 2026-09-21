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
 * An FX market for a Garman-Kohlhagen option.
 *
 * @param spot         current FX rate (domestic per foreign)
 * @param strike       option strike
 * @param volFx        lognormal FX volatility
 * @param rateDom      domestic rate
 * @param rateForeign  foreign rate
 */
@Of
public final class FxMarket {

  private FxMarket(double spot, double strike, double volFx, double rateDom, double rateForeign) {
    this.spot = spot;
    this.strike = strike;
    this.volFx = volFx;
    this.rateDom = rateDom;
    this.rateForeign = rateForeign;
  }

  private final double spot;
  private final double strike;
  private final double volFx;
  private final double rateDom;
  private final double rateForeign;
  static FxMarket create(double spot, double strike, double volFx, double rateDom, double rateForeign) {
    return new FxMarket(spot, strike, volFx, rateDom, rateForeign);
  }

  public static FxMarketBuilder of() {
    return new FxMarketBuilder();
  }

  public double spot() {
    return spot;
  }

  public double strike() {
    return strike;
  }

  public double volFx() {
    return volFx;
  }

  public double rateDom() {
    return rateDom;
  }

  public double rateForeign() {
    return rateForeign;
  }


  public FxMarket validated() {
    if (!(spot > 0 && strike > 0 && volFx >= 0)) {
      throw new IllegalArgumentException("invalid FX market: " + this);
    }
    return this;
  }

  public static FxMarket eurusd() {
    return FxMarket.of().spot(1.08).strike(1.10).volFx(0.09).rateDom(0.035).rateForeign(0.02).build();
  }
}
