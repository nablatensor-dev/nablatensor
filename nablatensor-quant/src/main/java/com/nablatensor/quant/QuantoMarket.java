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
 * A quanto market: a foreign asset and the FX rate it would convert through,
 * with their correlation. The payoff settles at a fixed FX rate, so only the
 * <em>drift adjustment</em> {@code -corr * volAsset * volFx} carries the FX.
 *
 * @param assetSpot    foreign-currency spot of the underlying
 * @param strike       strike in foreign-currency units
 * @param volAsset     lognormal vol of the foreign asset
 * @param volFx        lognormal vol of the FX rate
 * @param corr         asset/FX Brownian correlation
 * @param rateDom      domestic rate
 * @param rateForeign  foreign rate
 */
@Of
public final class QuantoMarket {

  private QuantoMarket(double assetSpot, double strike, double volAsset, double volFx, double corr, double rateDom, double rateForeign) {
    this.assetSpot = assetSpot;
    this.strike = strike;
    this.volAsset = volAsset;
    this.volFx = volFx;
    this.corr = corr;
    this.rateDom = rateDom;
    this.rateForeign = rateForeign;
  }

  private final double assetSpot;
  private final double strike;
  private final double volAsset;
  private final double volFx;
  private final double corr;
  private final double rateDom;
  private final double rateForeign;
  static QuantoMarket create(double assetSpot, double strike, double volAsset, double volFx, double corr, double rateDom, double rateForeign) {
    return new QuantoMarket(assetSpot, strike, volAsset, volFx, corr, rateDom, rateForeign);
  }

  public static QuantoMarketBuilder of() {
    return new QuantoMarketBuilder();
  }

  public double assetSpot() {
    return assetSpot;
  }

  public double strike() {
    return strike;
  }

  public double volAsset() {
    return volAsset;
  }

  public double volFx() {
    return volFx;
  }

  public double corr() {
    return corr;
  }

  public double rateDom() {
    return rateDom;
  }

  public double rateForeign() {
    return rateForeign;
  }


  public QuantoMarket validated() {
    if (!(assetSpot > 0 && strike > 0 && volAsset >= 0 && volFx >= 0 && corr > -1 && corr < 1)) {
      throw new IllegalArgumentException("invalid quanto market: " + this);
    }
    return this;
  }

  public static QuantoMarket base() {
    return QuantoMarket.of().assetSpot(100.0).strike(100.0).volAsset(0.25).volFx(0.10).corr(-0.3).rateDom(0.03).rateForeign(0.01).build();
  }
}
