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
package com.nablatensor.quant.analytic;

/**
 * Standard-normal density, distribution and quantile — the numerics the
 * analytic pricer oracles in this package share.
 *
 * <p>{@link #cdf} is West's rational approximation (double precision, absolute
 * error below {@code 1e-15}), not the lighter {@code erfc} approximation in
 * {@link com.nablatensor.quant.BlackScholes} — the oracle Greeks in this package
 * are taken by central differencing the closed forms, and a second difference
 * amplifies any bias in {@code N(x)} by {@code 1/h^2}. {@link #inverseCdf} is
 * the Acklam rational approximation refined by one Halley step against that
 * {@code cdf}, used where a formula needs {@code Phi^{-1}} — the Bachelier
 * implied strike, the Gaussian-copula default threshold.
 */
public final class Normal {

  private static final double SQRT_2PI = 2.506628274631000502;

  private Normal() {
  }

  /** Standard-normal probability density. */
  public static double pdf(double x) {
    return Math.exp(-0.5 * x * x) / SQRT_2PI;
  }

  /**
   * Standard-normal cumulative distribution, West's algorithm ("Better
   * approximations to cumulative normal functions", 2005): a seven-term rational
   * for {@code |x| < 7.07} and a continued fraction beyond it.
   */
  public static double cdf(double x) {
    double z = Math.abs(x);
    double c;
    if (z > 37.0) {
      c = 0.0;
    } else {
      double e = Math.exp(-0.5 * z * z);
      if (z < 7.07106781186547) {
        double num = 3.52624965998911e-02 * z + 0.700383064443688;
        num = num * z + 6.37396220353165;
        num = num * z + 33.912866078383;
        num = num * z + 112.079291497871;
        num = num * z + 221.213596169931;
        num = num * z + 220.206867912376;
        double den = 8.83883476483184e-02 * z + 1.75566716318264;
        den = den * z + 16.064177579207;
        den = den * z + 86.7807322029461;
        den = den * z + 296.564248779674;
        den = den * z + 637.333633378831;
        den = den * z + 793.826512519948;
        den = den * z + 440.413735824752;
        c = e * num / den;
      } else {
        double f = z + 0.65;
        f = z + 4.0 / f;
        f = z + 3.0 / f;
        f = z + 2.0 / f;
        f = z + 1.0 / f;
        c = e / f / SQRT_2PI;
      }
    }
    return x > 0.0 ? 1.0 - c : c;
  }

  /**
   * Standard-normal quantile {@code Phi^{-1}(p)} for {@code p} in {@code (0, 1)}.
   * Acklam's rational approximation with one Halley refinement against
   * {@link #cdf}.
   */
  public static double inverseCdf(double p) {
    if (!(p > 0.0 && p < 1.0)) {
      if (p == 0.0) {
        return Double.NEGATIVE_INFINITY;
      }
      if (p == 1.0) {
        return Double.POSITIVE_INFINITY;
      }
      throw new IllegalArgumentException("p must be in [0, 1], got " + p);
    }

    final double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    final double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01};
    final double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    final double[] d = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00};

    final double pLow = 0.02425;
    final double pHigh = 1.0 - pLow;
    double x;

    if (p < pLow) {
      double q = Math.sqrt(-2.0 * Math.log(p));
      x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
          / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= pHigh) {
      double q = p - 0.5;
      double r = q * q;
      x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
          / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
      double q = Math.sqrt(-2.0 * Math.log(1.0 - p));
      x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
          / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    double e = cdf(x) - p;
    double u = e * SQRT_2PI * Math.exp(0.5 * x * x);
    x = x - u / (1.0 + 0.5 * x * u);
    return x;
  }
}
