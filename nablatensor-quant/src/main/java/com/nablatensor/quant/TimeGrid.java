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
 * The simulation schedule for one path: how many steps, and what fraction of the
 * total maturity each step spans. Purely numeric — the maturity itself stays a
 * differentiable market input, so a model block scales each step by
 * {@code maturity * grid.fraction(i)} and theta still flows.
 *
 * <p>{@link #uniform(int)} is {@code n} equal steps and reproduces the earlier
 * {@code steps(int)} arithmetic exactly. {@link #of(double...)} takes explicit
 * ascending fixing times (the last is the maturity) and normalises the gaps, so
 * a path can be sampled densely near a barrier and sparsely elsewhere without
 * changing the model.
 */
public final class TimeGrid {

  private final double[] fraction;   // per-step share of maturity; sums to 1
  private final boolean uniform;

  private TimeGrid(double[] fraction, boolean uniform) {
    this.fraction = fraction;
    this.uniform = uniform;
  }

  /** {@code n} equally spaced steps. */
  public static TimeGrid uniform(int n) {
    if (n < 1) {
      throw new IllegalArgumentException("steps must be >= 1, got " + n);
    }
    double[] f = new double[n];
    Arrays.fill(f, 1.0 / n);
    return new TimeGrid(f, true);
  }

  /**
   * Explicit fixing times in ascending order; {@code times[times.length - 1]} is
   * the maturity. The returned grid has {@code times.length} steps whose
   * fractions are the normalised gaps {@code (t[i] - t[i-1]) / t[last]} with
   * {@code t[-1] = 0}.
   */
  public static TimeGrid of(double... times) {
    if (times.length < 1) {
      throw new IllegalArgumentException("need at least one fixing time");
    }
    double total = times[times.length - 1];
    if (!(total > 0.0)) {
      throw new IllegalArgumentException("the last fixing time must be positive, got " + total);
    }
    double[] f = new double[times.length];
    double prev = 0.0;
    for (int i = 0; i < times.length; i++) {
      double gap = times[i] - prev;
      if (!(gap > 0.0)) {
        throw new IllegalArgumentException("fixing times must be strictly ascending and positive");
      }
      f[i] = gap / total;
      prev = times[i];
    }
    boolean uniform = true;
    for (double v : f) {
      if (Math.abs(v - f[0]) > 1e-12) {
        uniform = false;
        break;
      }
    }
    return new TimeGrid(f, uniform);
  }

  /** Number of steps per path. */
  public int steps() {
    return fraction.length;
  }

  /** Step {@code i}'s share of the total maturity; {@code sum(fraction(i)) == 1}. */
  public double fraction(int i) {
    return fraction[i];
  }

  /** Cumulative share of maturity elapsed by the end of step {@code i}. */
  public double cumulative(int i) {
    double c = 0.0;
    for (int k = 0; k <= i; k++) {
      c += fraction[k];
    }
    return c;
  }

  /** Whether every step spans the same fraction (an {@link #uniform(int)} grid, or an equal-gap {@link #of}). */
  public boolean isUniform() {
    return uniform;
  }

  @Override
  public String toString() {
    return "TimeGrid(steps=" + fraction.length + (uniform ? ", uniform)" : ", non-uniform)");
  }
}
