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
 * A parametric local-volatility market. The reference spot the CEV ratio is
 * measured against is fixed at {@link #REF_SPOT}.
 *
 * @param spot   S0
 * @param strike K
 * @param rate   flat rate r
 * @param sigma0 volatility level at {@code S = REF_SPOT}
 * @param skew   CEV exponent on {@code S / REF_SPOT} (0 = GBM, negative = equity smile)
 */
public record LocalVolMarket(double spot, double strike, double rate, double sigma0, double skew) {

  /** The spot the local-vol ratio {@code (S / REF_SPOT)} is anchored to. */
  public static final double REF_SPOT = 100.0;

  public LocalVolMarket validated() {
    if (!(spot > 0 && strike > 0 && sigma0 > 0)) {
      throw new IllegalArgumentException("invalid local-vol market: " + this);
    }
    return this;
  }

  public static LocalVolMarket smile() {
    return new LocalVolMarket(100.0, 100.0, 0.02, 0.20, -0.5);
  }
}
