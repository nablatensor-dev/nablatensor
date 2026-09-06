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
 * A two-asset market for a spread or exchange option under correlated geometric
 * Brownian motion. Each asset has a spot, a lognormal volatility and a carry
 * (convenience) yield; the two share a flat rate. The Brownian correlation is a
 * fixed model input passed to {@link SpreadProducts}.
 *
 * @param s1     spot of the first leg
 * @param s2     spot of the second leg
 * @param vol1   lognormal vol of the first leg
 * @param vol2   lognormal vol of the second leg
 * @param yield1 carry / convenience yield on the first leg
 * @param yield2 carry / convenience yield on the second leg
 * @param rate   flat discount rate
 */
public record SpreadMarket(double s1, double s2, double vol1, double vol2,
                           double yield1, double yield2, double rate) {

  public SpreadMarket validated() {
    if (!(s1 > 0 && s2 > 0 && vol1 >= 0 && vol2 >= 0)) {
      throw new IllegalArgumentException("invalid spread market: " + this);
    }
    return this;
  }

  public static SpreadMarket sparkSpread() {
    // power vs gas, roughly: power ~ 60, gas-equivalent ~ 45, heat rate folded in.
    return new SpreadMarket(60.0, 45.0, 0.35, 0.30, 0.0, 0.0, 0.03);
  }
}
