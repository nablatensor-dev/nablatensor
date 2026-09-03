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
package com.nablatensor.cva;

import java.util.List;
import java.util.function.DoubleUnaryOperator;

/**
 * A counterparty credit curve as a piecewise-constant forward hazard rate
 * {@code lambda(t)} on a set of ascending tenor knots. Segment {@code i} covers
 * {@code (knot[i-1], knot[i]]} with {@code knot[-1] = 0}.
 *
 * <p>{@link #survival(double)} is {@code exp(-integral_0^t lambda)}; the default
 * probability over an interval is the drop in survival. The forward-hazard knots
 * are what a CVA valuation shocks for CS01, so they are exposed directly by
 * {@link #forwardHazards()} and shifted by {@link #withParallelShift(double)}.
 *
 * <p>{@link #bootstrap} strips a flat-forward hazard per segment from CDS par
 * quotes on a monthly premium/protection grid; {@link #fromFlatSpread} is the
 * one-segment {@code lambda = s / (1 - R)} approximation.
 */
public final class HazardCurve {

  private static final double BOOTSTRAP_STEP = 1.0 / 12.0;
  private static final double MAX_HAZARD = 5.0;

  private final double[] knotTimes;
  private final double[] forwardHazard;

  private HazardCurve(double[] knotTimes, double[] forwardHazard) {
    if (knotTimes.length != forwardHazard.length || knotTimes.length == 0) {
      throw new IllegalArgumentException("knot times and hazards must be non-empty and equal length");
    }
    double previous = 0.0;
    for (int i = 0; i < knotTimes.length; i++) {
      if (!(knotTimes[i] > previous)) {
        throw new IllegalArgumentException("knot times must be strictly ascending and positive");
      }
      if (!(forwardHazard[i] >= 0.0)) {
        throw new IllegalArgumentException("forward hazard must be >= 0, got " + forwardHazard[i]);
      }
      previous = knotTimes[i];
    }
    this.knotTimes = knotTimes.clone();
    this.forwardHazard = forwardHazard.clone();
  }

  public static HazardCurve piecewise(double[] knotTimes, double[] forwardHazard) {
    return new HazardCurve(knotTimes, forwardHazard);
  }

  /** A single flat forward hazard out to {@code lastTenor}. */
  public static HazardCurve flat(double hazard, double lastTenor) {
    return new HazardCurve(new double[] {lastTenor}, new double[] {hazard});
  }

  /** The {@code lambda = s / (1 - R)} approximation from one par spread. */
  public static HazardCurve fromFlatSpread(double parSpreadBp, double recovery, double lastTenor) {
    return flat(parSpreadBp * 1.0e-4 / (1.0 - recovery), lastTenor);
  }

  /**
   * Strips a flat-forward hazard per segment from ascending CDS par quotes, so
   * that the model par spread matches each quote given the earlier segments.
   * Premium and protection legs are integrated on a monthly grid; {@code discount}
   * maps a time in years to a risk-free discount factor.
   */
  public static HazardCurve bootstrap(List<CdsQuote> quotes, double recovery,
                                      DoubleUnaryOperator discount) {
    if (quotes.isEmpty()) {
      throw new IllegalArgumentException("need at least one CDS quote");
    }
    double lgd = 1.0 - recovery;
    int n = quotes.size();
    double[] knots = new double[n];
    double[] hazards = new double[n];
    double previousKnot = 0.0;
    for (int k = 0; k < n; k++) {
      CdsQuote quote = quotes.get(k);
      if (!(quote.tenorYears() > previousKnot)) {
        throw new IllegalArgumentException("CDS quotes must be in ascending tenor order");
      }
      knots[k] = quote.tenorYears();
      double target = quote.parSpread();
      int segment = k;
      double lo = 1.0e-6;
      double hi = MAX_HAZARD;
      for (int iteration = 0; iteration < 200; iteration++) {
        double mid = 0.5 * (lo + hi);
        hazards[segment] = mid;
        double modelSpread = parSpread(knots, hazards, segment + 1, lgd, discount);
        if (modelSpread > target) {
          hi = mid;
        } else {
          lo = mid;
        }
      }
      hazards[segment] = 0.5 * (lo + hi);
      previousKnot = knots[k];
    }
    return new HazardCurve(knots, hazards);
  }

  private static double parSpread(double[] knots, double[] hazards, int usableSegments,
                                  double lgd, DoubleUnaryOperator discount) {
    double maturity = knots[usableSegments - 1];
    double premium = 0.0;
    double protection = 0.0;
    double previousSurvival = 1.0;
    double previousTime = 0.0;
    for (double t = BOOTSTRAP_STEP; t <= maturity + 1.0e-9; t += BOOTSTRAP_STEP) {
      double survival = survivalFrom(knots, hazards, usableSegments, t);
      double df = discount.applyAsDouble(t);
      double accrual = t - previousTime;
      premium += df * accrual * 0.5 * (previousSurvival + survival);
      protection += lgd * df * (previousSurvival - survival);
      previousSurvival = survival;
      previousTime = t;
    }
    return premium == 0.0 ? 0.0 : protection / premium;
  }

  private static double survivalFrom(double[] knots, double[] hazards, int usableSegments, double t) {
    double cumulative = 0.0;
    double previous = 0.0;
    for (int i = 0; i < usableSegments; i++) {
      double upper = Math.min(t, knots[i]);
      if (upper > previous) {
        cumulative += hazards[i] * (upper - previous);
      }
      previous = knots[i];
      if (t <= knots[i]) {
        return Math.exp(-cumulative);
      }
    }
    // beyond the last usable knot: extend the last segment's hazard
    cumulative += hazards[usableSegments - 1] * (t - previous);
    return Math.exp(-cumulative);
  }

  public double hazardAt(double t) {
    for (int i = 0; i < knotTimes.length; i++) {
      if (t <= knotTimes[i]) {
        return forwardHazard[i];
      }
    }
    return forwardHazard[forwardHazard.length - 1];
  }

  public double cumulativeHazard(double t) {
    double cumulative = 0.0;
    double previous = 0.0;
    for (int i = 0; i < knotTimes.length; i++) {
      double upper = Math.min(t, knotTimes[i]);
      if (upper > previous) {
        cumulative += forwardHazard[i] * (upper - previous);
      }
      previous = knotTimes[i];
      if (t <= knotTimes[i]) {
        return cumulative;
      }
    }
    cumulative += forwardHazard[forwardHazard.length - 1] * (t - previous);
    return cumulative;
  }

  public double survival(double t) {
    return Math.exp(-cumulativeHazard(t));
  }

  public double defaultProbability(double from, double to) {
    return survival(from) - survival(to);
  }

  public double[] knotTimes() {
    return knotTimes.clone();
  }

  public double[] forwardHazards() {
    return forwardHazard.clone();
  }

  /** The same curve with a constant added to every forward-hazard segment (a CS01-style shift). */
  public HazardCurve withParallelShift(double deltaHazard) {
    double[] shifted = forwardHazard.clone();
    for (int i = 0; i < shifted.length; i++) {
      shifted[i] = Math.max(0.0, shifted[i] + deltaHazard);
    }
    return new HazardCurve(knotTimes, shifted);
  }

  /** A parallel CDS-spread bump translated to a hazard shift {@code d_lambda = d_s / (1 - R)}. */
  public HazardCurve shockedBySpread(double deltaSpreadBp, double recovery) {
    return withParallelShift(deltaSpreadBp * 1.0e-4 / (1.0 - recovery));
  }

  /** The piecewise-flat forward hazard sampled on {@code [0,2y], [2y,5y], [5y,+)} — the demo CS01 buckets. */
  public double[] threeBucketHazards() {
    return new double[] {hazardAt(1.0), hazardAt(3.5), hazardAt(7.5)};
  }
}
