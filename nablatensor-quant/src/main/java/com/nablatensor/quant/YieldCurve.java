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

import java.util.Arrays;

/**
 * A zero-coupon yield curve: continuously-compounded zero rates at ascending
 * pillar times, linearly interpolated in the zero rate and flat-extrapolated at
 * both ends.
 *
 * <p>Host-side infrastructure (plain {@code double}); the risk story is
 * {@link CurveBootstrap}, which produces one of these from par quotes together
 * with the analytic {@code d(zero) / d(quote)} Jacobian.
 */
public record YieldCurve(double[] pillars, double[] zeroRates) {

  public YieldCurve {
    if (pillars.length != zeroRates.length || pillars.length == 0) {
      throw new IllegalArgumentException("pillars and zeroRates must be non-empty and the same length");
    }
    for (int i = 1; i < pillars.length; i++) {
      if (!(pillars[i] > pillars[i - 1])) {
        throw new IllegalArgumentException("pillars must be strictly increasing");
      }
    }
    pillars = pillars.clone();
    zeroRates = zeroRates.clone();
  }

  /** Zero rate at {@code t}. */
  public double zeroRate(double t) {
    int n = pillars.length;
    if (t <= pillars[0]) {
      return zeroRates[0];
    }
    if (t >= pillars[n - 1]) {
      return zeroRates[n - 1];
    }
    int hi = 1;
    while (pillars[hi] < t) {
      hi++;
    }
    double w = (t - pillars[hi - 1]) / (pillars[hi] - pillars[hi - 1]);
    return zeroRates[hi - 1] + w * (zeroRates[hi] - zeroRates[hi - 1]);
  }

  /** Discount factor {@code P(0, t) = exp(-z(t) t)}. */
  public double discountFactor(double t) {
    return Math.exp(-zeroRate(t) * t);
  }

  /** Continuously-compounded forward rate between {@code t1} and {@code t2}. */
  public double forwardRate(double t1, double t2) {
    return (Math.log(discountFactor(t1)) - Math.log(discountFactor(t2))) / (t2 - t1);
  }

  /**
   * Par rate of a fixed-for-float swap with fixed-leg accruals {@code tau} at
   * ascending times {@code times} (times[i] paying tau[i]).
   */
  public double parSwapRate(double[] times, double[] tau) {
    double annuity = 0.0;
    for (int i = 0; i < times.length; i++) {
      annuity += tau[i] * discountFactor(times[i]);
    }
    return (1.0 - discountFactor(times[times.length - 1])) / annuity;
  }

  @Override
  public String toString() {
    return "YieldCurve" + Arrays.toString(pillars) + "=" + Arrays.toString(zeroRates);
  }
}
