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
package com.nablatensor.credit;

/**
 * A single-name credit curve as a piecewise-constant forward hazard rate on
 * ascending tenor knots. Segment {@code i} covers {@code (knot[i-1], knot[i]]}
 * with {@code knot[-1] = 0}; {@link #survival}{@code (t) = exp(-integral hazard)}.
 */
public record CreditCurve(double[] knotTimes, double[] forwardHazard) {

  public CreditCurve {
    if (knotTimes.length != forwardHazard.length || knotTimes.length == 0) {
      throw new IllegalArgumentException("knot times and hazards must be non-empty and equal length");
    }
    double prev = 0.0;
    for (int i = 0; i < knotTimes.length; i++) {
      if (!(knotTimes[i] > prev)) {
        throw new IllegalArgumentException("knot times must be strictly ascending and positive");
      }
      if (!(forwardHazard[i] >= 0.0)) {
        throw new IllegalArgumentException("forward hazard must be >= 0");
      }
      prev = knotTimes[i];
    }
    knotTimes = knotTimes.clone();
    forwardHazard = forwardHazard.clone();
  }

  @Override
  public double[] knotTimes() {
    return knotTimes.clone();
  }

  @Override
  public double[] forwardHazard() {
    return forwardHazard.clone();
  }

  /** A single flat forward hazard out to {@code lastTenor}. */
  public static CreditCurve flat(double hazard, double lastTenor) {
    return new CreditCurve(new double[] {lastTenor}, new double[] {hazard});
  }

  /** The {@code lambda = s / (1 - R)} approximation from one par CDS spread. */
  public static CreditCurve fromFlatSpread(double parSpread, double recovery, double lastTenor) {
    return flat(parSpread / (1.0 - recovery), lastTenor);
  }

  /** Cumulative hazard {@code integral_0^t lambda(u) du}. */
  public double cumulativeHazard(double t) {
    double h = 0.0;
    double prev = 0.0;
    for (int i = 0; i < knotTimes.length && prev < t; i++) {
      double seg = Math.min(t, knotTimes[i]) - prev;
      h += forwardHazard[i] * seg;
      prev = knotTimes[i];
    }
    if (t > prev) {
      h += forwardHazard[forwardHazard.length - 1] * (t - prev);
    }
    return h;
  }

  /** Survival probability {@code Q(0, t)}. */
  public double survival(double t) {
    return Math.exp(-cumulativeHazard(t));
  }

  /** Default probability by {@code t}. */
  public double defaultProbability(double t) {
    return 1.0 - survival(t);
  }

  /** A parallel shift of every forward hazard, for a CS01 bump. */
  public CreditCurve withParallelShift(double bump) {
    double[] h = forwardHazard.clone();
    for (int i = 0; i < h.length; i++) {
      h[i] = Math.max(0.0, h[i] + bump);
    }
    return new CreditCurve(knotTimes.clone(), h);
  }
}
