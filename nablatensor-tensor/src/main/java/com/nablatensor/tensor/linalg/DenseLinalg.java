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
package com.nablatensor.tensor.linalg;

/**
 * Reference dense factorizations for row-major {@code float[]} matrices:
 * Cholesky, LU with partial pivoting, Householder QR, symmetric eigen (cyclic
 * Jacobi) and thin SVD (one-sided Jacobi), plus the triangular solves built on
 * them.
 *
 * <p>These are the custom kernels behind {@link com.nablatensor.tensor.Linalg} — no
 * BLAS/LAPACK. They are <em>unblocked</em> (O(n&sup3;) with poor cache reuse
 * for large {@code n}); the point is that nablatensor has factorizations at all, on
 * every backend, via a host round-trip. Every accumulation is done in
 * {@code double} and rounded to {@code float} only on write, so the F32 results
 * are about as accurate as F32 can represent.
 */
public final class DenseLinalg {

  private DenseLinalg() {
  }

  // ---- Cholesky -------------------------------------------------------------

  /**
   * {@code A} (n&times;n, symmetric positive-definite) &rarr; lower-triangular
   * {@code L} (n&times;n) with {@code A = L·Lᵀ}. Only the lower triangle of
   * {@code A} is read.
   *
   * @throws IllegalArgumentException if a pivot is non-positive (not SPD)
   */
  public static float[] cholesky(float[] a, int n) {
    float[] l = new float[n * n];
    for (int j = 0; j < n; j++) {
      double d = a[j * n + j];
      for (int k = 0; k < j; k++) {
        double ljk = l[j * n + k];
        d -= ljk * ljk;
      }
      if (!(d > 0.0)) {
        throw new IllegalArgumentException(
            "matrix is not positive-definite: pivot " + j + " = " + d);
      }
      double ljj = Math.sqrt(d);
      l[j * n + j] = (float) ljj;
      for (int i = j + 1; i < n; i++) {
        double s = a[i * n + j];
        for (int k = 0; k < j; k++) {
          s -= (double) l[i * n + k] * l[j * n + k];
        }
        l[i * n + j] = (float) (s / ljj);
      }
    }
    return l;
  }

  // ---- LU with partial pivoting ------------------------------------------

  /**
   * LU factorization of {@code A} (n&times;n) with partial pivoting.
   *
   * @return {@code [packed, pivots]} — {@code packed} holds the unit-lower
   *     {@code L} strictly below the diagonal and {@code U} on and above it;
   *     {@code pivots[i]} is the original row index now at position {@code i}
   *     (so {@code P·A} has row {@code i} equal to row {@code pivots[i]} of
   *     {@code A}). Split {@code packed} with {@link #luLower}/{@link #luUpper}.
   */
  public static float[][] luDecompose(float[] a, int n) {
    double[] m = new double[n * n];
    for (int i = 0; i < m.length; i++) {
      m[i] = a[i];
    }
    int[] piv = new int[n];
    for (int i = 0; i < n; i++) {
      piv[i] = i;
    }
    for (int k = 0; k < n; k++) {
      int p = k;
      double best = Math.abs(m[k * n + k]);
      for (int i = k + 1; i < n; i++) {
        double v = Math.abs(m[i * n + k]);
        if (v > best) {
          best = v;
          p = i;
        }
      }
      if (p != k) {
        for (int j = 0; j < n; j++) {
          double t = m[k * n + j];
          m[k * n + j] = m[p * n + j];
          m[p * n + j] = t;
        }
        int t = piv[k];
        piv[k] = piv[p];
        piv[p] = t;
      }
      double pivot = m[k * n + k];
      if (pivot == 0.0) {
        continue; // singular column; leave it, callers detect via a zero U pivot
      }
      for (int i = k + 1; i < n; i++) {
        double factor = m[i * n + k] / pivot;
        m[i * n + k] = factor;
        for (int j = k + 1; j < n; j++) {
          m[i * n + j] -= factor * m[k * n + j];
        }
      }
    }
    float[] packed = new float[n * n];
    for (int i = 0; i < packed.length; i++) {
      packed[i] = (float) m[i];
    }
    float[] pivots = new float[n];
    for (int i = 0; i < n; i++) {
      pivots[i] = piv[i];
    }
    return new float[][] {packed, pivots};
  }

  /** Unit-lower {@code L} (1s on the diagonal) from a {@link #luDecompose} packed result. */
  public static float[] luLower(float[] packed, int n) {
    float[] l = new float[n * n];
    for (int i = 0; i < n; i++) {
      l[i * n + i] = 1f;
      for (int j = 0; j < i; j++) {
        l[i * n + j] = packed[i * n + j];
      }
    }
    return l;
  }

  /** Upper-triangular {@code U} from a {@link #luDecompose} packed result. */
  public static float[] luUpper(float[] packed, int n) {
    float[] u = new float[n * n];
    for (int i = 0; i < n; i++) {
      for (int j = i; j < n; j++) {
        u[i * n + j] = packed[i * n + j];
      }
    }
    return u;
  }

  // ---- Householder QR --------------------------------------------------

  /**
   * Full Householder QR of {@code A} (m&times;n), {@code m >= n}.
   *
   * @return {@code [Q, R]} — {@code Q} orthogonal (m&times;m), {@code R} upper
   *     (m&times;n), with {@code A = Q·R}.
   */
  public static float[][] qrDecompose(float[] a, int m, int n) {
    double[] r = new double[m * n];
    for (int i = 0; i < r.length; i++) {
      r[i] = a[i];
    }
    int steps = Math.min(m - 1, n);
    double[][] reflectors = new double[steps][]; // reflectors[k] has length m-k, or null
    double[] betas = new double[steps];
    double[] w = new double[Math.max(m, n)];     // reused row-buffer: w[j] = β·(vᵀM)[j]
    for (int k = 0; k < steps; k++) {
      double norm = 0.0;
      for (int i = k; i < m; i++) {
        double x = r[i * n + k];
        norm += x * x;
      }
      norm = Math.sqrt(norm);
      if (norm == 0.0) {
        continue;
      }
      double alpha = r[k * n + k] > 0.0 ? -norm : norm;
      int len = m - k;
      double[] v = new double[len];
      for (int i = 0; i < len; i++) {
        v[i] = r[(k + i) * n + k];
      }
      v[0] -= alpha;
      double vtv = 0.0;
      for (double vi : v) {
        vtv += vi * vi;
      }
      if (vtv == 0.0) {
        continue;
      }
      reflectors[k] = v;
      double beta = betas[k] = 2.0 / vtv;
      // R := (I - β·vvᵀ) R over columns k..n-1, unit-stride inner loops
      applyReflectorLeft(r, n, k, n, v, beta, w);
    }
    // Q = H₀···H_{steps-1}, built by applying the reflectors to I in reverse
    // over their trailing (m-k)×(m-k) blocks — O(m³/3), not O(n·m²).
    double[] q = new double[m * m];
    for (int i = 0; i < m; i++) {
      q[i * m + i] = 1.0;
    }
    for (int k = steps - 1; k >= 0; k--) {
      if (reflectors[k] != null) {
        applyReflectorLeft(q, m, k, m, reflectors[k], betas[k], w);
      }
    }
    float[] qf = new float[m * m];
    for (int i = 0; i < qf.length; i++) {
      qf[i] = (float) q[i];
    }
    float[] rf = new float[m * n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        rf[i * n + j] = j < i ? 0f : (float) r[i * n + j];
      }
    }
    return new float[][] {qf, rf};
  }

  /**
   * In place: {@code M[k:, k:cols) := (I - β·vvᵀ) M[k:, k:cols)} where {@code v}
   * has length {@code rows-k}. Both passes keep the column index innermost so
   * the row-major access is unit-stride. {@code w} is a scratch buffer of
   * length &ge; {@code cols}.
   */
  private static void applyReflectorLeft(double[] m, int stride, int k, int cols,
      double[] v, double beta, double[] w) {
    int len = v.length;
    for (int j = k; j < cols; j++) {
      w[j] = 0.0;
    }
    for (int i = 0; i < len; i++) {
      double vi = v[i];
      int base = (k + i) * stride;
      for (int j = k; j < cols; j++) {
        w[j] += vi * m[base + j];
      }
    }
    for (int j = k; j < cols; j++) {
      w[j] *= beta;
    }
    for (int i = 0; i < len; i++) {
      double vi = v[i];
      int base = (k + i) * stride;
      for (int j = k; j < cols; j++) {
        m[base + j] -= vi * w[j];
      }
    }
  }

  // ---- solves ---------------------------------------------------------

  /** Solve {@code A·X = B} for {@code X}; {@code A} (n&times;n), {@code B} (n&times;p), row-major. */
  public static float[] solve(float[] a, int n, float[] b, int p) {
    float[][] luPiv = luDecompose(a, n);
    return luSolve(luPiv[0], luPiv[1], n, b, p);
  }

  /** Solve using an existing {@link #luDecompose} result. */
  public static float[] luSolve(float[] packed, float[] pivots, int n, float[] b, int p) {
    double[] x = new double[n * p];
    for (int i = 0; i < n; i++) {
      int src = (int) pivots[i];
      for (int c = 0; c < p; c++) {
        x[i * p + c] = b[src * p + c];
      }
    }
    // forward: L·Y = P·B  (L unit-lower)
    for (int i = 0; i < n; i++) {
      for (int c = 0; c < p; c++) {
        double s = x[i * p + c];
        for (int k = 0; k < i; k++) {
          s -= (double) packed[i * n + k] * x[k * p + c];
        }
        x[i * p + c] = s;
      }
    }
    // back: U·X = Y
    for (int i = n - 1; i >= 0; i--) {
      double diag = packed[i * n + i];
      if (diag == 0.0) {
        throw new ArithmeticException("matrix is singular");
      }
      for (int c = 0; c < p; c++) {
        double s = x[i * p + c];
        for (int k = i + 1; k < n; k++) {
          s -= (double) packed[i * n + k] * x[k * p + c];
        }
        x[i * p + c] = s / diag;
      }
    }
    float[] out = new float[n * p];
    for (int i = 0; i < out.length; i++) {
      out[i] = (float) x[i];
    }
    return out;
  }

  /** {@code det(A)} via LU: product of the {@code U} diagonal times the permutation sign. */
  public static double det(float[] a, int n) {
    float[][] luPiv = luDecompose(a, n);
    float[] packed = luPiv[0];
    int[] piv = new int[n];
    for (int i = 0; i < n; i++) {
      piv[i] = (int) luPiv[1][i];
    }
    double d = 1.0;
    for (int i = 0; i < n; i++) {
      d *= packed[i * n + i];
    }
    boolean[] seen = new boolean[n];
    int transpositions = 0;
    for (int i = 0; i < n; i++) {
      if (seen[i]) {
        continue;
      }
      int j = i;
      int len = 0;
      while (!seen[j]) {
        seen[j] = true;
        j = piv[j];
        len++;
      }
      transpositions += len - 1;
    }
    return (transpositions & 1) == 0 ? d : -d;
  }

  // ---- symmetric eigendecomposition (tridiagonalize + QL/QR) ---------

  /**
   * Eigendecomposition of a symmetric {@code A} (n&times;n): Householder
   * reduction to tridiagonal form followed by the implicit-shift QL iteration
   * (the classic {@code tred2}/{@code tql2}). O(n&sup3;) once for the reduction
   * and the eigenvector accumulation, O(n&sup2;) per QL step — the same shape
   * as LAPACK's {@code ssyev}, not the O(sweeps·n&sup3;) of plain Jacobi.
   *
   * @return {@code [values, vectors]} — {@code values} length {@code n}
   *     ascending, {@code vectors} n&times;n with eigenvector {@code i} in
   *     column {@code i}, so {@code A = V·diag(values)·Vᵀ}.
   */
  public static float[][] eighSymmetric(float[] a, int n) {
    double[] z = new double[n * n]; // becomes the eigenvectors (columns)
    for (int i = 0; i < z.length; i++) {
      z[i] = a[i];
    }
    double[] d = new double[n];
    double[] e = new double[n];
    tred2(z, d, e, n);
    // transpose so tql2's Givens sweeps rotate *rows* (unit-stride), not columns
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        double t = z[i * n + j];
        z[i * n + j] = z[j * n + i];
        z[j * n + i] = t;
      }
    }
    tql2(z, d, e, n); // eigenvector k is now row k of z

    Integer[] order = new Integer[n];
    for (int i = 0; i < n; i++) {
      order[i] = i;
    }
    java.util.Arrays.sort(order, (x, y) -> Double.compare(d[x], d[y]));
    float[] values = new float[n];
    float[] vectors = new float[n * n];
    for (int col = 0; col < n; col++) {
      int src = order[col];
      values[col] = (float) d[src];
      for (int row = 0; row < n; row++) {
        vectors[row * n + col] = (float) z[src * n + row];
      }
    }
    return new float[][] {values, vectors};
  }

  /** Householder reduction of symmetric {@code z} to tridiagonal ({@code d}, {@code e}); {@code z} becomes the transform. */
  private static void tred2(double[] z, double[] d, double[] e, int n) {
    for (int i = n - 1; i >= 1; i--) {
      int l = i - 1;
      double h = 0.0;
      double scale = 0.0;
      if (l > 0) {
        for (int k = 0; k <= l; k++) {
          scale += Math.abs(z[i * n + k]);
        }
        if (scale == 0.0) {
          e[i] = z[i * n + l];
        } else {
          for (int k = 0; k <= l; k++) {
            z[i * n + k] /= scale;
            h += z[i * n + k] * z[i * n + k];
          }
          double f = z[i * n + l];
          double g = f >= 0.0 ? -Math.sqrt(h) : Math.sqrt(h);
          e[i] = scale * g;
          h -= f * g;
          z[i * n + l] = f - g;
          f = 0.0;
          for (int j = 0; j <= l; j++) {
            z[j * n + i] = z[i * n + j] / h;
            g = 0.0;
            for (int k = 0; k <= j; k++) {
              g += z[j * n + k] * z[i * n + k];
            }
            for (int k = j + 1; k <= l; k++) {
              g += z[k * n + j] * z[i * n + k];
            }
            e[j] = g / h;
            f += e[j] * z[i * n + j];
          }
          double hh = f / (h + h);
          for (int j = 0; j <= l; j++) {
            f = z[i * n + j];
            e[j] = g = e[j] - hh * f;
            for (int k = 0; k <= j; k++) {
              z[j * n + k] -= f * e[k] + g * z[i * n + k];
            }
          }
        }
      } else {
        e[i] = z[i * n + l];
      }
      d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
    double[] w = new double[n]; // w[j] = (z_iᵀ · Q)[j] — kept unit-stride
    for (int i = 0; i < n; i++) {
      int l = i - 1;
      if (d[i] != 0.0) {
        for (int j = 0; j <= l; j++) {
          w[j] = 0.0;
        }
        for (int k = 0; k <= l; k++) {
          double zik = z[i * n + k];
          int rk = k * n;
          for (int j = 0; j <= l; j++) {
            w[j] += zik * z[rk + j]; // unit stride in j
          }
        }
        for (int k = 0; k <= l; k++) {
          double zki = z[k * n + i]; // column i, O(n²) total
          int rk = k * n;
          for (int j = 0; j <= l; j++) {
            z[rk + j] -= zki * w[j]; // unit stride in j
          }
        }
      }
      d[i] = z[i * n + i];
      z[i * n + i] = 1.0;
      for (int j = 0; j <= l; j++) {
        z[j * n + i] = 0.0;
        z[i * n + j] = 0.0;
      }
    }
  }

  /** Implicit-shift QL iteration on tridiagonal ({@code d}, {@code e}); {@code z} accumulates the rotations. */
  private static void tql2(double[] z, double[] d, double[] e, int n) {
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
          if (dd + Math.abs(e[m]) == dd) {
            break; // e[m] is below the roundoff of its neighbours — deflated
          }
        }
        if (m != l) {
          if (iter++ == 50) {
            throw new ArithmeticException("eigh: QL failed to converge");
          }
          double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
          double r = pythag(g, 1.0);
          g = d[m] - d[l] + e[l] / (g + Math.copySign(r, g));
          double s = 1.0;
          double c = 1.0;
          double p = 0.0;
          for (int i = m - 1; i >= l; i--) {
            double f = s * e[i];
            double b = c * e[i];
            r = pythag(f, g);
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
            int r0 = i * n;
            int r1 = r0 + n;
            for (int k = 0; k < n; k++) {
              f = z[r1 + k];
              z[r1 + k] = s * z[r0 + k] + c * f;
              z[r0 + k] = c * z[r0 + k] - s * f;
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

  /** Overflow-safe {@code sqrt(a²+b²)} — the {@code Math.hypot} contract is stricter (and slower) than QL needs. */
  private static double pythag(double a, double b) {
    double absA = Math.abs(a);
    double absB = Math.abs(b);
    if (absA > absB) {
      double r = absB / absA;
      return absA * Math.sqrt(1.0 + r * r);
    }
    if (absB == 0.0) {
      return 0.0;
    }
    double r = absA / absB;
    return absB * Math.sqrt(1.0 + r * r);
  }

  // ---- thin SVD (via the eigendecomposition of AᵀA) -----------------

  /**
   * Thin SVD of {@code A} (m&times;n, {@code m >= n}) via the symmetric
   * eigendecomposition of {@code AᵀA = V·Σ²·Vᵀ}, then {@code U = A·V·Σ⁻¹}.
   *
   * <p>This reuses the fast {@link #eighSymmetric} and is far quicker than a
   * one-sided Jacobi SVD, at the cost of squaring the condition number — the
   * smallest singular values of a badly-scaled {@code A} lose about half their
   * digits. For a well-conditioned {@code A} the reconstruction is accurate to
   * F32.
   *
   * @return {@code [u, s, v]} — {@code u} m&times;n with orthonormal columns,
   *     {@code s} length {@code n} descending and non-negative, {@code v}
   *     n&times;n orthogonal, so {@code A = U·diag(s)·Vᵀ}.
   */
  public static float[][] svdThin(float[] a, int m, int n) {
    // G = AᵀA  (n×n, symmetric PSD): rank-1 updates from each row of A, so the
    // inner loops stay unit-stride instead of walking columns of A.
    double[] gd = new double[n * n];
    for (int k = 0; k < m; k++) {
      int rk = k * n;
      for (int i = 0; i < n; i++) {
        double aki = a[rk + i];
        if (aki == 0.0) {
          continue;
        }
        for (int j = i; j < n; j++) {
          gd[i * n + j] += aki * a[rk + j];
        }
      }
    }
    float[] g = new float[n * n];
    for (int i = 0; i < n; i++) {
      for (int j = i; j < n; j++) {
        float v = (float) gd[i * n + j];
        g[i * n + j] = v;
        g[j * n + i] = v;
      }
    }
    float[][] ev = eighSymmetric(g, n); // ascending
    float[] w = ev[0];
    float[] v = ev[1];

    float[] uf = new float[m * n];
    float[] sf = new float[n];
    float[] vf = new float[n * n];
    for (int col = 0; col < n; col++) {
      int src = n - 1 - col; // eigen ascending -> singular descending
      double sigma = Math.sqrt(Math.max(0.0, w[src]));
      sf[col] = (float) sigma;
      double inv = sigma > 0.0 ? 1.0 / sigma : 0.0;
      for (int row = 0; row < n; row++) {
        vf[row * n + col] = v[row * n + src];
      }
      for (int row = 0; row < m; row++) {
        double acc = 0.0;
        for (int k = 0; k < n; k++) {
          acc += (double) a[row * n + k] * v[k * n + src];
        }
        uf[row * n + col] = (float) (acc * inv);
      }
    }
    return new float[][] {uf, sf, vf};
  }
}
