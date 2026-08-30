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
package com.nablatensor.tensor;

import com.nablatensor.tensor.linalg.DenseLinalg;

/**
 * Dense linear-algebra factorizations and solves over {@link Tensor}:
 * {@link #cholesky}, {@link #lu}, {@link #qr}, {@link #solve}, {@link #inv},
 * {@link #det}.
 *
 * <p>These run through {@link DenseLinalg} — custom, unblocked, F32
 * factorizations, no BLAS/LAPACK. The tensor is brought to the host, factored,
 * and the results uploaded back to the <em>same device</em>, so every backend
 * (CPU, SIMD, CUDA, Vulkan, ROCm) gets factorizations even though none has a
 * native kernel for them. That host round-trip is O(n&sup2;) against the
 * O(n&sup3;) factorization, and factorizations are rarely on a hot path — but
 * do not put these inside an inner loop over a GPU-resident tensor.
 */
public final class Linalg {

  private Linalg() {
  }

  /** {@code A = L·Lᵀ} with unit permutation. */
  public record Lu(Tensor l, Tensor u, Tensor pivots) {
  }

  /** {@code A = Q·R}. */
  public record Qr(Tensor q, Tensor r) {
  }

  /** Symmetric eigendecomposition {@code A = V·diag(values)·Vᵀ}; {@code values} ascending, eigenvectors in columns of {@code vectors}. */
  public record Eigh(Tensor values, Tensor vectors) {
  }

  /** Thin SVD {@code A = U·diag(s)·Vᵀ}; {@code s} descending and non-negative. */
  public record Svd(Tensor u, Tensor s, Tensor v) {
  }

  /**
   * Cholesky factor of a symmetric positive-definite {@code A} (n&times;n):
   * lower-triangular {@code L} with {@code A = L·Lᵀ}.
   *
   * @throws IllegalArgumentException if {@code A} is not square or not SPD
   */
  public static Tensor cholesky(Tensor a) {
    int n = requireSquare(a, "cholesky");
    float[] l = DenseLinalg.cholesky(a.toFloatArray(), n);
    return NablaTensors.arrayOn(l, Shape.of(n, n), a.device());
  }

  /**
   * LU factorization with partial pivoting of {@code A} (n&times;n): returns
   * unit-lower {@code L}, upper {@code U}, and a length-{@code n} pivot vector
   * where {@code pivots[i]} is the original row now at position {@code i}
   * (so {@code L·U} equals {@code A} with its rows permuted by {@code pivots}).
   */
  public static Lu lu(Tensor a) {
    int n = requireSquare(a, "lu");
    float[][] packed = DenseLinalg.luDecompose(a.toFloatArray(), n);
    Tensor l = NablaTensors.arrayOn(DenseLinalg.luLower(packed[0], n), Shape.of(n, n), a.device());
    Tensor u = NablaTensors.arrayOn(DenseLinalg.luUpper(packed[0], n), Shape.of(n, n), a.device());
    Tensor pivots = NablaTensors.arrayOn(packed[1], Shape.of(n), a.device());
    return new Lu(l, u, pivots);
  }

  /**
   * Full Householder QR of {@code A} (m&times;n), {@code m >= n}: orthogonal
   * {@code Q} (m&times;m) and upper {@code R} (m&times;n) with {@code A = Q·R}.
   */
  public static Qr qr(Tensor a) {
    Shape s = a.shape();
    if (s.rank() != 2) {
      throw new IllegalArgumentException("qr expects a rank-2 tensor, got " + s);
    }
    int m = s.dim(0);
    int n = s.dim(1);
    if (m < n) {
      throw new IllegalArgumentException("qr requires rows >= cols, got " + s);
    }
    float[][] qr = DenseLinalg.qrDecompose(a.toFloatArray(), m, n);
    return new Qr(NablaTensors.arrayOn(qr[0], Shape.of(m, m), a.device()),
        NablaTensors.arrayOn(qr[1], Shape.of(m, n), a.device()));
  }

  /**
   * Eigendecomposition of a symmetric {@code A} (n&times;n) by cyclic Jacobi.
   * The input is symmetrized ({@code (A + Aᵀ)/2}) first, so only the symmetric
   * part is used. Returns eigenvalues ascending and the matching eigenvectors
   * in the columns of {@code vectors}, with {@code A = V·diag(values)·Vᵀ}.
   */
  public static Eigh eigh(Tensor a) {
    int n = requireSquare(a, "eigh");
    float[] m = a.toFloatArray();
    float[] sym = new float[n * n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        sym[i * n + j] = 0.5f * (m[i * n + j] + m[j * n + i]);
      }
    }
    float[][] ev = DenseLinalg.eighSymmetric(sym, n);
    return new Eigh(NablaTensors.arrayOn(ev[0], Shape.of(n), a.device()),
        NablaTensors.arrayOn(ev[1], Shape.of(n, n), a.device()));
  }

  /**
   * Thin SVD of {@code A} (m&times;n, {@code m >= n}) by one-sided Jacobi:
   * {@code U} (m&times;n, orthonormal columns), {@code s} (n, descending &ge; 0),
   * {@code V} (n&times;n, orthogonal), with {@code A = U·diag(s)·Vᵀ}.
   */
  public static Svd svd(Tensor a) {
    Shape shape = a.shape();
    if (shape.rank() != 2) {
      throw new IllegalArgumentException("svd expects a rank-2 tensor, got " + shape);
    }
    int m = shape.dim(0);
    int n = shape.dim(1);
    if (m < n) {
      throw new IllegalArgumentException("svd requires rows >= cols, got " + shape
          + " (transpose the input and swap U/V)");
    }
    float[][] usv = DenseLinalg.svdThin(a.toFloatArray(), m, n);
    return new Svd(NablaTensors.arrayOn(usv[0], Shape.of(m, n), a.device()),
        NablaTensors.arrayOn(usv[1], Shape.of(n), a.device()),
        NablaTensors.arrayOn(usv[2], Shape.of(n, n), a.device()));
  }

  /**
   * Solves {@code A·X = B} for {@code X} via LU with partial pivoting.
   * {@code A} is (n&times;n); {@code B} is a length-{@code n} vector or an
   * (n&times;p) matrix, and {@code X} has the same shape as {@code B}.
   */
  public static Tensor solve(Tensor a, Tensor b) {
    int n = requireSquare(a, "solve");
    Shape bs = b.shape();
    if (bs.rank() < 1 || bs.rank() > 2 || bs.dim(0) != n) {
      throw new IllegalArgumentException("solve: B must have " + n + " rows, got " + bs);
    }
    int p = bs.rank() == 1 ? 1 : bs.dim(1);
    float[] x = DenseLinalg.solve(a.toFloatArray(), n, b.toFloatArray(), p);
    return NablaTensors.arrayOn(x, bs, a.device());
  }

  /** The inverse of a square {@code A}, i.e. {@code solve(A, I)}. */
  public static Tensor inv(Tensor a) {
    int n = requireSquare(a, "inv");
    float[] identity = new float[n * n];
    for (int i = 0; i < n; i++) {
      identity[i * n + i] = 1f;
    }
    float[] x = DenseLinalg.solve(a.toFloatArray(), n, identity, n);
    return NablaTensors.arrayOn(x, Shape.of(n, n), a.device());
  }

  /** The determinant of a square {@code A}, computed from its LU factorization (in {@code double}). */
  public static double det(Tensor a) {
    int n = requireSquare(a, "det");
    return DenseLinalg.det(a.toFloatArray(), n);
  }

  private static int requireSquare(Tensor a, String op) {
    Shape s = a.shape();
    if (s.rank() != 2 || s.dim(0) != s.dim(1)) {
      throw new IllegalArgumentException(op + " expects a square rank-2 tensor, got " + s);
    }
    return s.dim(0);
  }
}
