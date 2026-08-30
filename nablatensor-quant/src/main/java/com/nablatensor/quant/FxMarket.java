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
 * An FX market for a Garman-Kohlhagen option.
 *
 * @param spot         current FX rate (domestic per foreign)
 * @param strike       option strike
 * @param volFx        lognormal FX volatility
 * @param rateDom      domestic rate
 * @param rateForeign  foreign rate
 */
public record FxMarket(double spot, double strike, double volFx, double rateDom, double rateForeign) {

  public FxMarket validated() {
    if (!(spot > 0 && strike > 0 && volFx >= 0)) {
      throw new IllegalArgumentException("invalid FX market: " + this);
    }
    return this;
  }

  public static FxMarket eurusd() {
    return new FxMarket(1.08, 1.10, 0.09, 0.035, 0.02);
  }
}
