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
 * Principal-component decomposition of a symmetric covariance or correlation
 * matrix by the cyclic Jacobi eigenvalue algorithm — the standard first step in
 * a term-structure factor analysis (level / slope / curvature) and in reducing
 * a large correlation matrix to a handful of driving factors.
 *
 * <p>{@link #eigenvalues} are in descending order; column {@code i} of
 * {@link #loadings} is the unit eigenvector for {@code eigenvalues[i]};
 * {@link #explainedVariance}{@code [i]} is that eigenvalue as a fraction of the
 * trace.
 */
public record Pca(double[] eigenvalues, double[][] loadings, double[] explainedVariance) {

  public Pca {
    eigenvalues = eigenvalues.clone();
    loadings = deepCopy(loadings);
    explainedVariance = explainedVariance.clone();
  }

  @Override
  public double[] eigenvalues() {
    return eigenvalues.clone();
  }

  @Override
  public double[][] loadings() {
    return deepCopy(loadings);
  }

  @Override
  public double[] explainedVariance() {
    return explainedVariance.clone();
  }

  public static Pca of(double[][] symmetric) {
    int n = symmetric.length;
    double[][] a = deepCopy(symmetric);
    double[][] v = identity(n);

    for (int sweep = 0; sweep < 100; sweep++) {
      double off = 0.0;
      for (int p = 0; p < n; p++) {
        for (int q = p + 1; q < n; q++) {
          off += a[p][q] * a[p][q];
        }
      }
      if (off < 1e-30) {
        break;
      }
      for (int p = 0; p < n; p++) {
        for (int q = p + 1; q < n; q++) {
          if (Math.abs(a[p][q]) < 1e-300) {
            continue;
          }
          double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
          double t = Math.signum(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1.0));
          if (theta == 0.0) {
            t = 1.0;
          }
          double c = 1.0 / Math.sqrt(t * t + 1.0);
          double s = t * c;
          rotate(a, v, p, q, c, s, n);
        }
      }
    }

    double[] eig = new double[n];
    for (int i = 0; i < n; i++) {
      eig[i] = a[i][i];
    }

    Integer[] order = new Integer[n];
    for (int i = 0; i < n; i++) {
      order[i] = i;
    }
    java.util.Arrays.sort(order, (x, y) -> Double.compare(eig[y], eig[x]));

    double[] sortedEig = new double[n];
    double[][] sortedVec = new double[n][n];
    double trace = 0.0;
    for (double e : eig) {
      trace += e;
    }
    double[] explained = new double[n];
    for (int col = 0; col < n; col++) {
      int src = order[col];
      sortedEig[col] = eig[src];
      explained[col] = trace == 0.0 ? 0.0 : eig[src] / trace;
      for (int row = 0; row < n; row++) {
        sortedVec[row][col] = v[row][src];
      }
    }
    return new Pca(sortedEig, sortedVec, explained);
  }

  private static void rotate(double[][] a, double[][] v, int p, int q, double c, double s, int n) {
    for (int k = 0; k < n; k++) {
      double akp = a[k][p];
      double akq = a[k][q];
      a[k][p] = c * akp - s * akq;
      a[k][q] = s * akp + c * akq;
    }
    for (int k = 0; k < n; k++) {
      double apk = a[p][k];
      double aqk = a[q][k];
      a[p][k] = c * apk - s * aqk;
      a[q][k] = s * apk + c * aqk;
    }
    for (int k = 0; k < n; k++) {
      double vkp = v[k][p];
      double vkq = v[k][q];
      v[k][p] = c * vkp - s * vkq;
      v[k][q] = s * vkp + c * vkq;
    }
  }

  private static double[][] identity(int n) {
    double[][] m = new double[n][n];
    for (int i = 0; i < n; i++) {
      m[i][i] = 1.0;
    }
    return m;
  }

  private static double[][] deepCopy(double[][] m) {
    double[][] c = new double[m.length][];
    for (int i = 0; i < m.length; i++) {
      c[i] = m[i].clone();
    }
    return c;
  }
}
