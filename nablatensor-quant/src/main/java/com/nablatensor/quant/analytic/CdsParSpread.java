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
package com.nablatensor.quant.analytic;

/**
 * The par spread of a single-name credit default swap from a survival curve and
 * a discount curve on a common time grid — the closed-form reference for the
 * bootstrapped-hazard CDS pricing in {@code nablatensor-cva}.
 *
 * <p>The grid {@code times[i]} (ascending, all {@code > 0}) carries the survival
 * probability {@code survival[i] = Q(0, times[i])} and the risk-free discount
 * factor {@code discount[i] = P(0, times[i])}; time zero has {@code Q = 1},
 * {@code P = 1} implicitly. Over segment {@code i = (times[i-1], times[i]]} of
 * length {@code dt_i}:
 *
 * <pre>{@code
 * protection leg  = (1 - R) * sum_i  Dmid_i * (Q_{i-1} - Q_i)
 * premium annuity = sum_i  dt_i * D_i * (Q_{i-1} + Q_i) / 2      (accrual on default, half period)
 * par spread      = protection leg / premium annuity
 * }</pre>
 *
 * where {@code Dmid_i = (D_{i-1} + D_i) / 2} approximates discounting to the
 * mid-point of the segment in which default occurs. A monthly (or finer) grid
 * keeps the piecewise-flat approximation tight.
 */
public record CdsParSpread(double parSpread, double protectionLeg, double premiumAnnuity) {

  /**
   * @param recovery recovery rate {@code R} in {@code [0, 1)}; loss given default is {@code 1 - R}
   * @param times    ascending segment end times in years, all {@code > 0}
   * @param survival {@code Q(0, times[i])}, non-increasing, first entry {@code <= 1}
   * @param discount {@code P(0, times[i])}, positive
   */
  public static CdsParSpread of(double recovery, double[] times, double[] survival, double[] discount) {
    int n = times.length;
    if (n == 0 || survival.length != n || discount.length != n) {
      throw new IllegalArgumentException("times, survival and discount must be non-empty and the same length");
    }
    if (!(recovery >= 0.0 && recovery < 1.0)) {
      throw new IllegalArgumentException("recovery must be in [0, 1), got " + recovery);
    }

    double lgd = 1.0 - recovery;
    double protection = 0.0;
    double annuity = 0.0;

    double prevT = 0.0;
    double prevQ = 1.0;
    double prevD = 1.0;
    for (int i = 0; i < n; i++) {
      double tt = times[i];
      double q = survival[i];
      double dd = discount[i];
      if (!(tt > prevT)) {
        throw new IllegalArgumentException("times must be strictly ascending and positive");
      }
      double dt = tt - prevT;
      double dMid = 0.5 * (prevD + dd);

      protection += lgd * dMid * (prevQ - q);
      annuity += dt * dd * 0.5 * (prevQ + q);

      prevT = tt;
      prevQ = q;
      prevD = dd;
    }

    return new CdsParSpread(protection / annuity, protection, annuity);
  }

  /** Mark-to-market of a bought-protection position paying {@code contractSpread} on unit notional. */
  public double protectionBuyerValue(double contractSpread) {
    return protectionLeg - contractSpread * premiumAnnuity;
  }
}
