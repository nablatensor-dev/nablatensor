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

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A post-LIBOR curve stack: one OIS <em>discount</em> curve plus one or more
 * tenor-indexed <em>forecast</em> curves. Cash flows are always discounted on
 * {@link #discount()}; floating-leg forwards come off the forecast curve of the
 * relevant tenor, so a fixed-for-float swap is only at par when the two curves
 * differ — that gap is the tenor basis.
 *
 * <p>Produced by {@link MultiCurveBootstrap}, which also returns the full
 * {@code d(zero rate) / d(quote)} Jacobian across both curves (a forecast zero
 * rate depends on the OIS quotes through the discount factors in its par-swap
 * equation).
 *
 * <p>This is the stylised annual construction textbooks use to introduce OIS
 * discounting: fixed and floating legs share an annual grid, so a swap of
 * maturity {@code N} adds exactly one forecast pillar. Sub-annual float
 * frequencies and interpolation inside the solve are a later refinement.
 */
public record CurveSet(YieldCurve discount, Map<String, YieldCurve> forecast) {

  public CurveSet {
    forecast = Map.copyOf(forecast);
  }

  public YieldCurve forecast(String tenor) {
    YieldCurve c = forecast.get(tenor);
    if (c == null) {
      throw new IllegalArgumentException("no forecast curve for tenor '" + tenor + "'; have " + forecast.keySet());
    }
    return c;
  }

  /** Discount factor {@code P_d(0, t)} off the OIS curve. */
  public double df(double t) {
    return discount.discountFactor(t);
  }

  /** Simple-compounded forward {@code (P_fc(t1)/P_fc(t2) - 1) / (t2 - t1)} off a forecast curve. */
  public double forwardRate(String tenor, double t1, double t2) {
    YieldCurve fc = forecast(tenor);
    return (fc.discountFactor(t1) / fc.discountFactor(t2) - 1.0) / (t2 - t1);
  }

  /** Annual fixed-leg annuity {@code sum_{i=1}^{N} P_d(i)}. */
  public double annuity(int maturityYears) {
    double a = 0.0;
    for (int i = 1; i <= maturityYears; i++) {
      a += discount.discountFactor(i);
    }
    return a;
  }

  /**
   * Par rate of the stylised annual multi-curve swap of maturity {@code N}:
   * {@code sum_i (P_fc(i-1)/P_fc(i) - 1) P_d(i)  /  sum_i P_d(i)}.
   */
  public double parSwapRate(String tenor, int maturityYears) {
    YieldCurve fc = forecast(tenor);
    double floatPv = 0.0;
    double annuity = 0.0;
    double prevFc = 1.0;
    for (int i = 1; i <= maturityYears; i++) {
      double pd = discount.discountFactor(i);
      double pfc = fc.discountFactor(i);
      floatPv += (prevFc / pfc - 1.0) * pd;
      annuity += pd;
      prevFc = pfc;
    }
    return floatPv / annuity;
  }

  /** The tenor labels present, for iteration. */
  public java.util.Set<String> tenors() {
    return forecast.keySet();
  }

  @Override
  public String toString() {
    Map<String, Object> m = new LinkedHashMap<>();
    m.put("discount", discount);
    forecast.forEach(m::put);
    return "CurveSet" + m;
  }
}
