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

import com.nablatensor.engine.SDouble;
import com.nablatensor.quant.Calibrator;

/**
 * The exponentially weighted moving-average volatility estimator:
 *
 * <pre>{@code
 * sigma2_t = lambda sigma2_{t-1} + (1 - lambda) r_{t-1}^2
 * }</pre>
 *
 * <p>It is the {@code omega = 0}, {@code alpha + beta = 1} corner of
 * {@link Garch11} — no mean reversion, so the forecast variance is flat. The
 * RiskMetrics convention fixes {@code lambda = 0.94} for daily data;
 * {@link #estimateByMaximumLikelihood} instead fits it, driven by the same
 * adjoint-gradient optimiser as every other calibration in the library.
 */
public final class Ewma {

  private Ewma() {
  }

  /**
   * The conditional-variance path for a given decay, seeded with the sample
   * variance of {@code returns}. {@code variance[t]} is the estimate that
   * stands <em>before</em> observing {@code returns[t]}.
   */
  public static double[] conditionalVariance(double[] returns, double lambda) {
    if (!(lambda > 0.0 && lambda < 1.0)) {
      throw new IllegalArgumentException("lambda must be in (0, 1), got " + lambda);
    }
    double[] v = new double[returns.length];
    double s2 = sampleVariance(returns);
    for (int t = 0; t < returns.length; t++) {
      v[t] = s2;
      s2 = lambda * s2 + (1.0 - lambda) * returns[t] * returns[t];
    }
    return v;
  }

  /** Gaussian negative log-likelihood (up to a constant) of {@code returns} under EWMA. */
  public static double negLogLikelihood(double[] returns, double lambda) {
    double s2 = sampleVariance(returns);
    double nll = 0.0;
    for (double r : returns) {
      nll += Math.log(s2) + r * r / s2;
      s2 = lambda * s2 + (1.0 - lambda) * r * r;
    }
    return 0.5 * nll;
  }

  /**
   * Maximum-likelihood decay {@code lambda in [0.5, 0.9999]}. One scalar
   * parameter, but the objective is still recorded and the search still uses
   * the exact adjoint derivative — this is the smallest illustration of
   * "the adjoint gradient is the score vector".
   */
  public static double estimateByMaximumLikelihood(double[] returns) {
    double var0 = sampleVariance(returns);
    Calibrator.Result r = Calibrator.of(rec -> {
      SDouble lambda = rec.input("lambda", 0.94);
      SDouble s2 = rec.constant(var0);
      SDouble nll = rec.constant(0.0);
      for (double ret : returns) {
        SDouble r2 = rec.constant(ret * ret);
        nll = nll.add(s2.log()).add(r2.div(s2));
        s2 = lambda.mul(s2).add(rec.constant(1.0).sub(lambda).mul(r2));
      }
      rec.output(nll.mul(0.5));
    }).parameter("lambda", 0.94, 0.5, 0.9999).maxIterations(200)
        .on("cpu")   // tape has one node per observation; use the scalar replay
        .solve();
    return r.parameters().get("lambda");
  }

  static double sampleVariance(double[] x) {
    double mean = 0.0;
    for (double v : x) {
      mean += v;
    }
    mean /= x.length;
    double s = 0.0;
    for (double v : x) {
      double d = v - mean;
      s += d * d;
    }
    return s / Math.max(1, x.length - 1);
  }
}
