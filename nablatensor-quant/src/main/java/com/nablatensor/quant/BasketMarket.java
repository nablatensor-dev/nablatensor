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
 * A three-asset basket market: one spot and one lognormal vol per asset, plus a
 * shared flat rate. Correlation and weights are passed to
 * {@link BasketOption#option} as fixed model inputs.
 */
public record BasketMarket(double s1, double s2, double s3,
                           double v1, double v2, double v3, double rate) {

  public BasketMarket validated() {
    if (!(s1 > 0 && s2 > 0 && s3 > 0 && v1 >= 0 && v2 >= 0 && v3 >= 0)) {
      throw new IllegalArgumentException("invalid basket market: " + this);
    }
    return this;
  }

  public static BasketMarket equalWeighted() {
    return new BasketMarket(100.0, 100.0, 100.0, 0.20, 0.25, 0.30, 0.02);
  }
}
