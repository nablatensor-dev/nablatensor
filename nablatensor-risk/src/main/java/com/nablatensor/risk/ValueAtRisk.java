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
import com.nablatensor.quant.estimate.Pca;
import java.util.List;

/**
 * Value at Risk and Expected Shortfall by the three standard routes:
 *
 * <ul>
 *   <li><b>Historical / full-revaluation</b> — an empirical quantile of a
 *       {@link PnlVector}. Feed it a bootstrapped-scenario revaluation and this
 *       is full-revaluation VaR; feed it a window of realised daily P&L and it
 *       is historical-simulation VaR.</li>
 *   <li><b>Delta-normal</b> — {@code z_alpha * sqrt(s' Sigma s)}, the
 *       variance-covariance method, exact for a linear portfolio and a Gaussian
 *       risk-factor move.</li>
 *   <li><b>Delta-gamma Cornish-Fisher</b> — the quadratic P&L
 *       {@code delta' x + 0.5 x' Gamma x} has a skewed, fat-tailed distribution;
 *       its first four cumulants are computed in closed form and the quantile
 *       comes from a Cornish-Fisher expansion.</li>
 * </ul>
 *
 * <p>Every result is a <b>positive loss</b> quoted at confidence {@code alpha}
 * (e.g. {@code 0.99}). Multi-day figures use the square-root-of-time scaling the
 * caller supplies via {@code horizonDays} (a one-day covariance times
 * {@code sqrt(horizonDays)}).
 */
public final class ValueAtRisk {

  private ValueAtRisk() {
  }

  // ---- historical / full revaluation -------------------------------------

  /** Empirical VaR: the {@code alpha}-quantile of the loss sample, linearly interpolated. */
  public static double historical(PnlVector sample, double alpha) {
    checkAlpha(alpha);
    double[] loss = sample.sortedLosses();
    double rank = alpha * (loss.length - 1);
    int lo = (int) Math.floor(rank);
    int hi = Math.min(lo + 1, loss.length - 1);
    double frac = rank - lo;
    return loss[lo] + frac * (loss[hi] - loss[lo]);
  }

  /** Empirical Expected Shortfall: the mean loss in the {@code (1 - alpha)} tail beyond VaR. */
  public static double expectedShortfall(PnlVector sample, double alpha) {
    checkAlpha(alpha);
    double[] loss = sample.sortedLosses();
    int n = loss.length;
    int tail = Math.max(1, (int) Math.ceil((1.0 - alpha) * n));
    double s = 0.0;
    for (int i = n - tail; i < n; i++) {
      s += loss[i];
    }
    return s / tail;
  }

  // ---- delta-normal ----------------------------------------------------

  /**
   * @param sensitivities dP/dx per risk factor (the adjoint gradient)
   * @param covariance    one-period risk-factor covariance matrix, same order
   * @param alpha         confidence, e.g. {@code 0.99}
   * @param horizonDays   holding period; the covariance is scaled by {@code sqrt(horizonDays)}
   */
  public static double deltaNormal(double[] sensitivities, double[][] covariance,
                                   double alpha, double horizonDays) {
    checkAlpha(alpha);
    int n = sensitivities.length;
    double variance = 0.0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        variance += sensitivities[i] * covariance[i][j] * sensitivities[j];
      }
    }
    double sd = Math.sqrt(Math.max(variance, 0.0)) * Math.sqrt(horizonDays);
    return Normal.inverseCdf(alpha) * sd;
  }

  /** Delta-normal from a {@link Sensitivities} vector and an explicit factor order. */
  public static double deltaNormal(Sensitivities sensitivities, List<RiskFactor> order,
                                   double[][] covariance, double alpha, double horizonDays) {
    double[] s = new double[order.size()];
    for (int i = 0; i < s.length; i++) {
      s[i] = sensitivities.get(order.get(i));
    }
    return deltaNormal(s, covariance, alpha, horizonDays);
  }

  // ---- delta-gamma Cornish-Fisher ------------------------------------

  /** The first four cumulants of the quadratic P&L, and the moment shape. */
  public record QuadraticCumulants(double mean, double variance, double skewness, double excessKurtosis) {}

  /**
   * Cumulants of {@code Q = delta' x + 0.5 x' Gamma x} for {@code x ~ N(0, Sigma)}.
   * Diagonalises {@code Sigma^{1/2} Gamma Sigma^{1/2}} so {@code Q} becomes a sum
   * of independent {@code a_i y_i + 0.5 b_i y_i^2} terms with {@code y_i ~ N(0,1)}.
   */
  public static QuadraticCumulants quadraticCumulants(double[] delta, double[][] gamma, double[][] sigma) {
    int n = delta.length;
    double[][] sqrtSigma = matrixSqrt(sigma);
    double[][] m = mul(mul(sqrtSigma, gamma), sqrtSigma);      // Sigma^{1/2} Gamma Sigma^{1/2}
    Pca eig = Pca.of(m);
    double[] b = eig.eigenvalues();                             // b_i
    double[][] u = eig.loadings();

    // a = U' Sigma^{1/2} delta
    double[] sd = matVec(sqrtSigma, delta);
    double[] a = new double[n];
    for (int i = 0; i < n; i++) {
      double s = 0.0;
      for (int k = 0; k < n; k++) {
        s += u[k][i] * sd[k];
      }
      a[i] = s;
    }

    double k1 = 0.0;
    double k2 = 0.0;
    double k3 = 0.0;
    double k4 = 0.0;
    for (int i = 0; i < n; i++) {
      double ai = a[i];
      double bi = b[i];
      k1 += 0.5 * bi;
      k2 += ai * ai + 0.5 * bi * bi;
      k3 += 3.0 * ai * ai * bi + bi * bi * bi;
      k4 += 12.0 * ai * ai * bi * bi + 3.0 * bi * bi * bi * bi;
    }
    double skew = k2 <= 0.0 ? 0.0 : k3 / Math.pow(k2, 1.5);
    double kurt = k2 <= 0.0 ? 0.0 : k4 / (k2 * k2);
    return new QuadraticCumulants(k1, k2, skew, kurt);
  }

  /**
   * Delta-gamma VaR via a fourth-order Cornish-Fisher expansion of the loss
   * quantile.
   */
  public static double deltaGammaCornishFisher(double[] delta, double[][] gamma, double[][] sigma,
                                               double alpha, double horizonDays) {
    checkAlpha(alpha);
    QuadraticCumulants c = quadraticCumulants(delta, gamma, sigma);
    double z = Normal.inverseCdf(1.0 - alpha);                  // lower-tail P&L quantile, negative
    double s = c.skewness();
    double k = c.excessKurtosis();
    double w = z
        + (z * z - 1.0) / 6.0 * s
        + (z * z * z - 3.0 * z) / 24.0 * k
        - (2.0 * z * z * z - 5.0 * z) / 36.0 * s * s;
    double pnlQuantile = c.mean() + Math.sqrt(Math.max(c.variance(), 0.0)) * w;
    // horizon scaling on the standard deviation of the move
    double scaled = c.mean() + (pnlQuantile - c.mean()) * Math.sqrt(horizonDays);
    return -scaled;
  }

  // ---- linear algebra helpers -----------------------------------------

  private static double[][] matrixSqrt(double[][] symmetric) {
    Pca e = Pca.of(symmetric);
    double[] lam = e.eigenvalues();
    double[][] v = e.loadings();
    int n = lam.length;
    double[][] r = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        double s = 0.0;
        for (int k = 0; k < n; k++) {
          s += v[i][k] * Math.sqrt(Math.max(lam[k], 0.0)) * v[j][k];
        }
        r[i][j] = s;
      }
    }
    return r;
  }

  private static double[][] mul(double[][] a, double[][] b) {
    int n = a.length;
    int m = b[0].length;
    int p = b.length;
    double[][] r = new double[n][m];
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < p; k++) {
        double aik = a[i][k];
        if (aik == 0.0) {
          continue;
        }
        for (int j = 0; j < m; j++) {
          r[i][j] += aik * b[k][j];
        }
      }
    }
    return r;
  }

  private static double[] matVec(double[][] a, double[] x) {
    int n = a.length;
    double[] r = new double[n];
    for (int i = 0; i < n; i++) {
      double s = 0.0;
      for (int j = 0; j < x.length; j++) {
        s += a[i][j] * x[j];
      }
      r[i] = s;
    }
    return r;
  }

  private static void checkAlpha(double alpha) {
    if (!(alpha > 0.0 && alpha < 1.0)) {
      throw new IllegalArgumentException("alpha (confidence) must be in (0, 1), got " + alpha);
    }
  }
}
