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
 * A four-forward LMM market: the strip {@code L1..L4}, a flat instantaneous
 * volatility and a flat Brownian correlation. Accrual period length is fixed at
 * {@link #TENOR}.
 */
public record LmmMarket(double l1, double l2, double l3, double l4, double vol, double corr) {

  /** Accrual-period length of every forward in the strip, in years. */
  public static final double TENOR = 0.5;

  public LmmMarket validated() {
    if (!(l1 > 0 && l2 > 0 && l3 > 0 && l4 > 0 && vol > 0 && corr > -1 && corr <= 1)) {
      throw new IllegalArgumentException("invalid LMM market: " + this);
    }
    return this;
  }

  public static LmmMarket flat3pct() {
    return new LmmMarket(0.03, 0.03, 0.03, 0.03, 0.25, 0.8);
  }
}
