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
package com.nablatensor.quant.transform;

import java.util.Arrays;
import java.util.function.ToDoubleFunction;

/**
 * A plain n-dimensional Nelder-Mead simplex minimiser, used to calibrate the
 * COS-priced models (whose objective is not recordable — complex arithmetic and
 * a cosine series). Box bounds are enforced by returning a large penalty for an
 * out-of-range point.
 */
final class NelderMead {

  record Result(double[] x, double value, int iterations, boolean converged) {}

  private NelderMead() {
  }

  static Result minimise(ToDoubleFunction<double[]> f, double[] start, double[] step,
                         double[] lo, double[] hi, int maxIter, double tol) {
    int n = start.length;
    double[][] p = new double[n + 1][];
    double[] fv = new double[n + 1];
    p[0] = clamp(start.clone(), lo, hi);
    fv[0] = f.applyAsDouble(p[0]);
    for (int i = 0; i < n; i++) {
      double[] v = start.clone();
      v[i] += step[i];
      p[i + 1] = clamp(v, lo, hi);
      fv[i + 1] = f.applyAsDouble(p[i + 1]);
    }

    int iter = 0;
    for (; iter < maxIter; iter++) {
      order(p, fv);
      if (Math.abs(fv[n] - fv[0]) <= tol * (Math.abs(fv[0]) + tol)) {
        break;
      }
      double[] centroid = new double[n];
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          centroid[j] += p[i][j] / n;
        }
      }
      double[] worst = p[n];
      double[] reflected = combine(centroid, worst, 1.0, lo, hi);
      double fr = f.applyAsDouble(reflected);
      if (fr < fv[0]) {
        double[] expanded = combine(centroid, worst, 2.0, lo, hi);
        double fe = f.applyAsDouble(expanded);
        replaceWorst(p, fv, fe < fr ? expanded : reflected, Math.min(fe, fr));
      } else if (fr < fv[n - 1]) {
        replaceWorst(p, fv, reflected, fr);
      } else {
        double[] contracted = combine(centroid, worst, -0.5, lo, hi);
        double fc = f.applyAsDouble(contracted);
        if (fc < fv[n]) {
          replaceWorst(p, fv, contracted, fc);
        } else {
          for (int i = 1; i <= n; i++) {
            for (int j = 0; j < n; j++) {
              p[i][j] = p[0][j] + 0.5 * (p[i][j] - p[0][j]);
            }
            p[i] = clamp(p[i], lo, hi);
            fv[i] = f.applyAsDouble(p[i]);
          }
        }
      }
    }
    order(p, fv);
    return new Result(p[0].clone(), fv[0], iter, iter < maxIter);
  }

  private static double[] combine(double[] centroid, double[] worst, double coef, double[] lo, double[] hi) {
    double[] out = new double[centroid.length];
    for (int i = 0; i < out.length; i++) {
      out[i] = centroid[i] + coef * (centroid[i] - worst[i]);
    }
    return clamp(out, lo, hi);
  }

  private static void replaceWorst(double[][] p, double[] fv, double[] point, double value) {
    p[p.length - 1] = point;
    fv[fv.length - 1] = value;
  }

  private static void order(double[][] p, double[] fv) {
    Integer[] idx = new Integer[fv.length];
    for (int i = 0; i < idx.length; i++) {
      idx[i] = i;
    }
    Arrays.sort(idx, (x, y) -> Double.compare(fv[x], fv[y]));
    double[][] np = new double[p.length][];
    double[] nfv = new double[fv.length];
    for (int i = 0; i < idx.length; i++) {
      np[i] = p[idx[i]];
      nfv[i] = fv[idx[i]];
    }
    System.arraycopy(np, 0, p, 0, p.length);
    System.arraycopy(nfv, 0, fv, 0, fv.length);
  }

  private static double[] clamp(double[] x, double[] lo, double[] hi) {
    for (int i = 0; i < x.length; i++) {
      x[i] = Math.max(lo[i], Math.min(hi[i], x[i]));
    }
    return x;
  }
}
