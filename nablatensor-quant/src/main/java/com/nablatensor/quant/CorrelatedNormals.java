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

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.SDouble;

/**
 * Turns independent standard-normal draws into correlated ones by a fixed
 * Cholesky factor. The correlation matrix is a plain {@code double[][]} (a model
 * parameter, not a risk factor), so the factor is computed once at record time
 * and the per-step mixing is a handful of primitive nodes.
 */
public final class CorrelatedNormals {

  private final double[][] lower;

  private CorrelatedNormals(double[][] lower) {
    this.lower = lower;
  }

  /** Lower-triangular Cholesky factor {@code L} with {@code L Lᵀ = corr}. */
  public static CorrelatedNormals of(double[][] corr) {
    int n = corr.length;
    double[][] l = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++) {
        double sum = corr[i][j];
        for (int k = 0; k < j; k++) {
          sum -= l[i][k] * l[j][k];
        }
        if (i == j) {
          if (sum <= 0.0) {
            throw new IllegalArgumentException("correlation matrix is not positive definite at index " + i);
          }
          l[i][j] = Math.sqrt(sum);
        } else {
          l[i][j] = sum / l[j][j];
        }
      }
    }
    return new CorrelatedNormals(l);
  }

  /** The 2×2 case: {@code [[1, rho], [rho, 1]]}. */
  public static CorrelatedNormals pair(double rho) {
    return of(new double[][] {{1.0, rho}, {rho, 1.0}});
  }

  public int size() {
    return lower.length;
  }

  /**
   * Mixes {@code size()} independent draws into {@code size()} correlated ones:
   * {@code y = L x}.
   */
  public SDouble[] mix(AadRecorder rec, SDouble[] independent) {
    if (independent.length != lower.length) {
      throw new IllegalArgumentException("need " + lower.length + " draws, got " + independent.length);
    }
    SDouble[] y = new SDouble[lower.length];
    for (int i = 0; i < lower.length; i++) {
      SDouble acc = rec.constant(0.0);
      for (int j = 0; j <= i; j++) {
        acc = acc.add(independent[j].mul(lower[i][j]));
      }
      y[i] = acc;
    }
    return y;
  }

  /** Draws {@code size()} fresh independent normals and mixes them. */
  public SDouble[] draw(AadRecorder rec) {
    SDouble[] z = new SDouble[lower.length];
    for (int i = 0; i < z.length; i++) {
      z[i] = rec.randn();
    }
    return mix(rec, z);
  }
}
