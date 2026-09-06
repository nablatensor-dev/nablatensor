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
package com.nablatensor.quant.estimate;

/**
 * Sample and exponentially weighted covariance / correlation matrices for a
 * panel of return series — the multivariate companions of {@link Ewma}, and the
 * input {@link Pca} decomposes.
 *
 * <p>{@code returns[k]} is the {@code k}-th series (all the same length); the
 * result is {@code n x n} for {@code n} series. Series are treated as
 * zero-mean for the EWMA estimator (the RiskMetrics convention) and
 * demeaned for the sample estimator.
 */
public final class CorrelationEstimator {

  private CorrelationEstimator() {
  }

  /** Demeaned sample covariance, divisor {@code T - 1}. */
  public static double[][] sampleCovariance(double[][] returns) {
    int n = returns.length;
    int t = returns[0].length;
    double[] mean = new double[n];
    for (int i = 0; i < n; i++) {
      for (double v : returns[i]) {
        mean[i] += v;
      }
      mean[i] /= t;
    }
    double[][] cov = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = i; j < n; j++) {
        double s = 0.0;
        for (int k = 0; k < t; k++) {
          s += (returns[i][k] - mean[i]) * (returns[j][k] - mean[j]);
        }
        cov[i][j] = cov[j][i] = s / (t - 1);
      }
    }
    return cov;
  }

  /** Correlation matrix from a covariance matrix. */
  public static double[][] toCorrelation(double[][] cov) {
    int n = cov.length;
    double[] sd = new double[n];
    for (int i = 0; i < n; i++) {
      sd[i] = Math.sqrt(cov[i][i]);
    }
    double[][] corr = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        corr[i][j] = cov[i][j] / (sd[i] * sd[j]);
      }
    }
    return corr;
  }

  /** Demeaned sample correlation. */
  public static double[][] sampleCorrelation(double[][] returns) {
    return toCorrelation(sampleCovariance(returns));
  }

  /**
   * EWMA covariance with decay {@code lambda}, seeded with the sample
   * covariance and rolled forward over all observations (zero-mean series).
   */
  public static double[][] ewmaCovariance(double[][] returns, double lambda) {
    if (!(lambda > 0.0 && lambda < 1.0)) {
      throw new IllegalArgumentException("lambda must be in (0, 1), got " + lambda);
    }
    int n = returns.length;
    int t = returns[0].length;
    double[][] cov = sampleCovariance(returns);
    for (int k = 0; k < t; k++) {
      for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
          double updated = lambda * cov[i][j] + (1.0 - lambda) * returns[i][k] * returns[j][k];
          cov[i][j] = cov[j][i] = updated;
        }
      }
    }
    return cov;
  }

  /** EWMA correlation with decay {@code lambda}. */
  public static double[][] ewmaCorrelation(double[][] returns, double lambda) {
    return toCorrelation(ewmaCovariance(returns, lambda));
  }
}
