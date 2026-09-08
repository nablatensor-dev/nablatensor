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
 * The portfolio loss distribution under the one-factor Gaussian copula, by the
 * Andersen-Sidenius-Basu recursion: conditional on the systemic factor the
 * per-name default indicators are independent, so the number of defaults is a
 * Poisson-binomial whose distribution is built by convolving one name in at a
 * time; the unconditional distribution is a Gauss-Hermite integral over the
 * factor.
 *
 * <p>This build assumes a homogeneous loss given default so losses fall on an
 * integer grid of "one defaulted name". Heterogeneous notionals or recoveries
 * are a bucketed extension of the same recursion.
 */
public final class PortfolioLossDistribution {

  private final double[] pmf;        // P(k names defaulted), k = 0 .. n
  private final double lgd;
  private final int names;

  private PortfolioLossDistribution(double[] pmf, double lgd, int names) {
    this.pmf = pmf;
    this.lgd = lgd;
    this.names = names;
  }

  /**
   * @param pd    unconditional default probability per name to the horizon (same for every name)
   * @param names pool size
   * @param rho   copula correlation
   * @param lgd   loss given default, as a fraction of one name's notional
   * @param nodes Gauss-Hermite nodes for the factor integral (e.g. 64)
   */
  public static PortfolioLossDistribution homogeneous(double pd, int names, double rho,
                                                      double lgd, int nodes) {
    double[] x = new double[nodes];
    double[] w = new double[nodes];
    gaussHermite(nodes, x, w);

    double[] pmf = new double[names + 1];
    double norm = 1.0 / Math.sqrt(Math.PI);
    for (int g = 0; g < nodes; g++) {
      double m = Math.sqrt(2.0) * x[g];                       // change of variable to N(0,1)
      double p = OneFactorGaussianCopula.conditionalDefaultProbability(pd, rho, m);
      double[] cond = binomialPmf(names, p);
      double weight = w[g] * norm;
      for (int k = 0; k <= names; k++) {
        pmf[k] += weight * cond[k];
      }
    }
    return new PortfolioLossDistribution(pmf, lgd, names);
  }

  /** {@code P(k names defaulted)}. */
  public double probabilityOfDefaults(int k) {
    return (k < 0 || k > names) ? 0.0 : pmf[k];
  }

  /** Expected fractional portfolio loss. */
  public double expectedLoss() {
    double e = 0.0;
    for (int k = 0; k <= names; k++) {
      e += pmf[k] * (k * lgd / names);
    }
    return e;
  }

  /**
   * Expected loss of the tranche {@code [attach, detach]} (as fractions of the
   * portfolio), i.e. {@code E[ min(max(L - attach, 0), detach - attach) ]}.
   */
  public double expectedTrancheLoss(double attach, double detach) {
    double width = detach - attach;
    double e = 0.0;
    for (int k = 0; k <= names; k++) {
      double portfolioLoss = k * lgd / names;
      double trancheLoss = Math.min(Math.max(portfolioLoss - attach, 0.0), width);
      e += pmf[k] * trancheLoss;
    }
    return e;
  }

  // ---- helpers ------------------------------------------------------

  private static double[] binomialPmf(int n, double p) {
    // Convolve n Bernoulli(p) — the ASB recursion for the homogeneous case.
    double[] dist = new double[n + 1];
    dist[0] = 1.0;
    for (int i = 0; i < n; i++) {
      for (int k = i + 1; k >= 1; k--) {
        dist[k] = dist[k] * (1.0 - p) + dist[k - 1] * p;
      }
      dist[0] *= (1.0 - p);
    }
    return dist;
  }

  /** Gauss-Hermite abscissae and weights (physicists' convention, weight {@code e^{-x^2}}). */
  static void gaussHermite(int n, double[] x, double[] w) {
    // Golub-Welsch on the Hermite three-term recurrence.
    double[] diag = new double[n];
    double[] off = new double[n];
    for (int i = 1; i < n; i++) {
      off[i] = Math.sqrt(i / 2.0);
    }
    double[][] z = new double[n][n];
    for (int i = 0; i < n; i++) {
      z[i][i] = 1.0;
    }
    tqli(diag, off, n, z);
    Integer[] order = new Integer[n];
    for (int i = 0; i < n; i++) {
      order[i] = i;
    }
    java.util.Arrays.sort(order, (a, b) -> Double.compare(diag[a], diag[b]));
    double sqrtPi = Math.sqrt(Math.PI);
    for (int i = 0; i < n; i++) {
      int s = order[i];
      x[i] = diag[s];
      w[i] = sqrtPi * z[0][s] * z[0][s];
    }
  }

  /** Symmetric tridiagonal QL with implicit shifts (Numerical Recipes style). */
  private static void tqli(double[] d, double[] e, int n, double[][] zed) {
    for (int i = 1; i < n; i++) {
      e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    for (int l = 0; l < n; l++) {
      int iter = 0;
      int m;
      do {
        for (m = l; m < n - 1; m++) {
          double dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
          if (Math.abs(e[m]) <= 1e-15 * dd) {
            break;
          }
        }
        if (m != l) {
          if (iter++ == 60) {
            throw new IllegalStateException("tqli: no convergence");
          }
          double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
          double r = Math.hypot(g, 1.0);
          g = d[m] - d[l] + e[l] / (g + Math.copySign(r, g));
          double s = 1.0;
          double c = 1.0;
          double p = 0.0;
          for (int i = m - 1; i >= l; i--) {
            double f = s * e[i];
            double b = c * e[i];
            r = Math.hypot(f, g);
            e[i + 1] = r;
            if (r == 0.0) {
              d[i + 1] -= p;
              e[m] = 0.0;
              break;
            }
            s = f / r;
            c = g / r;
            g = d[i + 1] - p;
            r = (d[i] - g) * s + 2.0 * c * b;
            p = s * r;
            d[i + 1] = g + p;
            g = c * r - b;
            for (int k = 0; k < n; k++) {
              f = zed[k][i + 1];
              zed[k][i + 1] = s * zed[k][i] + c * f;
              zed[k][i] = c * zed[k][i] - s * f;
            }
          }
          if (r == 0.0 && (m - 1) >= l) {
            continue;
          }
          d[l] -= p;
          e[l] = g;
          e[m] = 0.0;
        }
      } while (m != l);
    }
  }
}
