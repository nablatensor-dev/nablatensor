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

import com.nablatensor.quant.analytic.Normal;

/**
 * Coverage tests for a one-day VaR forecast against a realised P&L series: the
 * count of exceptions (a loss worse than the forecast), Kupiec's unconditional
 * proportion-of-failures test, and Christoffersen's independence test plus their
 * combined conditional-coverage statistic.
 *
 * <p>The reference distributions are chi-square with one and two degrees of
 * freedom, whose survival functions are closed forms — {@code 2 (1 - Phi(sqrt s))}
 * for one degree, {@code exp(-s/2)} for two — so no incomplete-gamma routine is
 * needed.
 */
public record VarBacktest(int observations, int exceptions, double expectedExceptions,
                          double kupiecStatistic, double kupiecPValue,
                          double christoffersenStatistic, double christoffersenPValue,
                          double conditionalCoverageStatistic, double conditionalCoveragePValue) {

  /**
   * @param realisedPnl   realised P&L, one per day (loss is {@code -pnl})
   * @param varForecast   the VaR forecast for each day (positive loss numbers, same length)
   * @param alpha         the confidence the forecast was made at, e.g. {@code 0.99}
   */
  public static VarBacktest of(double[] realisedPnl, double[] varForecast, double alpha) {
    if (realisedPnl.length != varForecast.length || realisedPnl.length == 0) {
      throw new IllegalArgumentException("realisedPnl and varForecast must be non-empty and the same length");
    }
    if (!(alpha > 0.0 && alpha < 1.0)) {
      throw new IllegalArgumentException("alpha must be in (0, 1)");
    }
    int n = realisedPnl.length;
    double p = 1.0 - alpha;                          // expected exception probability

    boolean[] hit = new boolean[n];
    int x = 0;
    for (int t = 0; t < n; t++) {
      hit[t] = -realisedPnl[t] > varForecast[t];
      if (hit[t]) {
        x++;
      }
    }

    double kupiec = kupiecPof(n, x, p);
    double kupiecP = chiSquareSurvival1(kupiec);
    double ind = christoffersenIndependence(hit);
    double indP = chiSquareSurvival1(ind);
    double cc = kupiec + ind;
    double ccP = chiSquareSurvival2(cc);

    return new VarBacktest(n, x, n * p, kupiec, kupiecP, ind, indP, cc, ccP);
  }

  /** Constant-forecast convenience. */
  public static VarBacktest of(double[] realisedPnl, double varForecast, double alpha) {
    double[] f = new double[realisedPnl.length];
    java.util.Arrays.fill(f, varForecast);
    return of(realisedPnl, f, alpha);
  }

  /** True if the model is rejected at the given significance (e.g. {@code 0.05}) by conditional coverage. */
  public boolean rejectedAt(double significance) {
    return conditionalCoveragePValue < significance;
  }

  // ---- statistics -----------------------------------------------------

  private static double kupiecPof(int n, int x, double p) {
    if (x == 0) {
      return -2.0 * n * Math.log(1.0 - p);
    }
    if (x == n) {
      return -2.0 * n * Math.log(p);
    }
    double piHat = (double) x / n;
    double logL0 = (n - x) * Math.log(1.0 - p) + x * Math.log(p);
    double logL1 = (n - x) * Math.log(1.0 - piHat) + x * Math.log(piHat);
    return -2.0 * (logL0 - logL1);
  }

  private static double christoffersenIndependence(boolean[] hit) {
    int n00 = 0;
    int n01 = 0;
    int n10 = 0;
    int n11 = 0;
    for (int t = 1; t < hit.length; t++) {
      boolean prev = hit[t - 1];
      boolean cur = hit[t];
      if (!prev && !cur) {
        n00++;
      } else if (!prev) {
        n01++;
      } else if (!cur) {
        n10++;
      } else {
        n11++;
      }
    }
    int n0 = n00 + n01;
    int n1 = n10 + n11;
    int total = n0 + n1;
    if (total == 0 || (n01 + n11) == 0 || (n01 + n11) == total) {
      return 0.0;                                    // no exceptions, or degenerate: nothing to test
    }
    double pi = (double) (n01 + n11) / total;
    double pi0 = n0 == 0 ? 0.0 : (double) n01 / n0;
    double pi1 = n1 == 0 ? 0.0 : (double) n11 / n1;

    double logLPooled = xlogy(n01 + n11, pi) + xlogy(n00 + n10, 1.0 - pi);
    double logLSplit = xlogy(n01, pi0) + xlogy(n00, 1.0 - pi0)
        + xlogy(n11, pi1) + xlogy(n10, 1.0 - pi1);
    return Math.max(0.0, -2.0 * (logLPooled - logLSplit));
  }

  private static double xlogy(int count, double prob) {
    return count == 0 ? 0.0 : count * Math.log(prob);
  }

  /** P(chi^2_1 > s) = 2 (1 - Phi(sqrt s)). */
  private static double chiSquareSurvival1(double s) {
    if (s <= 0.0) {
      return 1.0;
    }
    return 2.0 * (1.0 - Normal.cdf(Math.sqrt(s)));
  }

  /** P(chi^2_2 > s) = exp(-s/2). */
  private static double chiSquareSurvival2(double s) {
    return s <= 0.0 ? 1.0 : Math.exp(-0.5 * s);
  }
}
