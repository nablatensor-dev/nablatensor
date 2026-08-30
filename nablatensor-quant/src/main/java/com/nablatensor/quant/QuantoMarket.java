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
public record QuantoMarket(double assetSpot, double strike, double volAsset, double volFx,
                           double corr, double rateDom, double rateForeign) {

  public QuantoMarket validated() {
    if (!(assetSpot > 0 && strike > 0 && volAsset >= 0 && volFx >= 0 && corr > -1 && corr < 1)) {
      throw new IllegalArgumentException("invalid quanto market: " + this);
    }
    return this;
  }

  public static QuantoMarket base() {
    return new QuantoMarket(100.0, 100.0, 0.25, 0.10, -0.3, 0.03, 0.01);
  }
}
