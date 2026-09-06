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

/**
 * A deterministic annual seasonality function for a commodity log-forward curve:
 * a sum of harmonics of the yearly cycle,
 *
 * <pre>{@code
 * f(t) = sum_{k=1}^{K} ( aCos_k cos(2 pi k t) + aSin_k sin(2 pi k t) )
 * }</pre>
 *
 * with {@code t} in years. Add {@code f(t)} to a model's log-forward; the
 * coefficients are fitted once to the observed forward curve on the host and
 * then held fixed (not differentiated).
 */
public record Seasonality(double[] aCos, double[] aSin) {

  public Seasonality {
    if (aCos.length != aSin.length) {
      throw new IllegalArgumentException("aCos and aSin must have the same length");
    }
    aCos = aCos.clone();
    aSin = aSin.clone();
  }

  public int harmonics() {
    return aCos.length;
  }

  public double value(double t) {
    double s = 0.0;
    for (int k = 0; k < aCos.length; k++) {
      double w = 2.0 * Math.PI * (k + 1) * t;
      s += aCos[k] * Math.cos(w) + aSin[k] * Math.sin(w);
    }
    return s;
  }

  /**
   * Least-squares fit of {@code harmonics} harmonics to {@code (times, values)}
   * by the normal equations. {@code values} are the seasonal residuals of the
   * log-forward curve (curve minus its smooth trend).
   */
  public static Seasonality fit(double[] times, double[] values, int harmonics) {
    if (times.length != values.length) {
      throw new IllegalArgumentException("times and values must be the same length");
    }
    int p = 2 * harmonics;
    double[][] ata = new double[p][p];
    double[] atb = new double[p];
    for (int n = 0; n < times.length; n++) {
      double[] row = new double[p];
      for (int k = 0; k < harmonics; k++) {
        double w = 2.0 * Math.PI * (k + 1) * times[n];
        row[2 * k] = Math.cos(w);
        row[2 * k + 1] = Math.sin(w);
      }
      for (int i = 0; i < p; i++) {
        atb[i] += row[i] * values[n];
        for (int j = 0; j < p; j++) {
          ata[i][j] += row[i] * row[j];
        }
      }
    }
    double[] coef = solveSpd(ata, atb);
    double[] c = new double[harmonics];
    double[] s = new double[harmonics];
    for (int k = 0; k < harmonics; k++) {
      c[k] = coef[2 * k];
      s[k] = coef[2 * k + 1];
    }
    return new Seasonality(c, s);
  }

  private static double[] solveSpd(double[][] a, double[] b) {
    int n = b.length;
    double[][] l = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++) {
        double sum = a[i][j];
        for (int k = 0; k < j; k++) {
          sum -= l[i][k] * l[j][k];
        }
        l[i][j] = i == j ? Math.sqrt(Math.max(sum, 1e-18)) : sum / l[j][j];
      }
    }
    double[] y = new double[n];
    for (int i = 0; i < n; i++) {
      double sum = b[i];
      for (int k = 0; k < i; k++) {
        sum -= l[i][k] * y[k];
      }
      y[i] = sum / l[i][i];
    }
    double[] x = new double[n];
    for (int i = n - 1; i >= 0; i--) {
      double sum = y[i];
      for (int k = i + 1; k < n; k++) {
        sum -= l[k][i] * x[k];
      }
      x[i] = sum / l[i][i];
    }
    return x;
  }
}
