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
 * A three-asset basket market: one spot and one lognormal vol per asset, plus a
 * shared flat rate. Correlation and weights are passed to
 * {@link BasketOption#option} as fixed model inputs.
 */
@Of
public final class BasketMarket {

  private BasketMarket(double s1, double s2, double s3, double v1, double v2, double v3, double rate) {
    this.s1 = s1;
    this.s2 = s2;
    this.s3 = s3;
    this.v1 = v1;
    this.v2 = v2;
    this.v3 = v3;
    this.rate = rate;
  }

  private final double s1;
  private final double s2;
  private final double s3;
  private final double v1;
  private final double v2;
  private final double v3;
  private final double rate;
  static BasketMarket create(double s1, double s2, double s3, double v1, double v2, double v3, double rate) {
    return new BasketMarket(s1, s2, s3, v1, v2, v3, rate);
  }

  public static BasketMarketBuilder of() {
    return new BasketMarketBuilder();
  }

  public double s1() {
    return s1;
  }

  public double s2() {
    return s2;
  }

  public double s3() {
    return s3;
  }

  public double v1() {
    return v1;
  }

  public double v2() {
    return v2;
  }

  public double v3() {
    return v3;
  }

  public double rate() {
    return rate;
  }


  public BasketMarket validated() {
    if (!(s1 > 0 && s2 > 0 && s3 > 0 && v1 >= 0 && v2 >= 0 && v3 >= 0)) {
      throw new IllegalArgumentException("invalid basket market: " + this);
    }
    return this;
  }

  public static BasketMarket equalWeighted() {
    return BasketMarket.of().s1(100.0).s2(100.0).s3(100.0).v1(0.20).v2(0.25).v3(0.30).rate(0.02).build();
  }
}
