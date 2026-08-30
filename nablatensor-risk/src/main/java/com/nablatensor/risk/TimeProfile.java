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
package com.nablatensor.risk;

/**
 * An exposure (or any quantity) sampled on a time grid — the shape a
 * netting-set XVA profile takes. Kept minimal here; XVA valuation on top of it
 * is the rest of Phase 2.
 *
 * @param times  ascending time points, years
 * @param values the quantity at each time
 */
public record TimeProfile(double[] times, double[] values) {

  public TimeProfile {
    if (times.length != values.length || times.length == 0) {
      throw new IllegalArgumentException("times and values must be non-empty and the same length");
    }
    times = times.clone();
    values = values.clone();
  }

  /** Expected positive exposure: time-average of {@code max(value, 0)}. */
  public double epe() {
    double s = 0;
    for (double v : values) {
      s += Math.max(v, 0.0);
    }
    return s / values.length;
  }

  /** Expected negative exposure: time-average of {@code min(value, 0)}. */
  public double ene() {
    double s = 0;
    for (double v : values) {
      s += Math.min(v, 0.0);
    }
    return s / values.length;
  }

  public double peak() {
    double m = Double.NEGATIVE_INFINITY;
    for (double v : values) {
      m = Math.max(m, v);
    }
    return m;
  }

  /** Trapezoidal {@code integral(discount(t) * max(value,0) * hazard(t) dt)} — a CVA-style reduction. */
  public double weightedIntegral(java.util.function.DoubleUnaryOperator weight) {
    double acc = 0;
    for (int i = 1; i < times.length; i++) {
      double dt = times[i] - times[i - 1];
      double a = Math.max(values[i - 1], 0.0) * weight.applyAsDouble(times[i - 1]);
      double b = Math.max(values[i], 0.0) * weight.applyAsDouble(times[i]);
      acc += 0.5 * (a + b) * dt;
    }
    return acc;
  }
}
