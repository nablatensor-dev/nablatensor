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
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A GARCH(1,1) conditional-variance model,
 *
 * <pre>{@code
 * sigma2_t = omega + alpha r_{t-1}^2 + beta sigma2_{t-1}
 * }</pre>
 *
 * fitted by Gaussian maximum likelihood. The negative log-likelihood is
 * <em>recorded</em> once and every optimiser iteration reads the exact score
 * vector from a single adjoint sweep — the same machinery a SABR or Heston
 * calibration uses, applied to a time series instead of an option surface.
 *
 * <p>Stationarity ({@code omega > 0}, {@code alpha, beta >= 0},
 * {@code alpha + beta < 1}) is enforced by fitting in the reparameterised
 * coordinates {@code (omega, persistence = alpha + beta, share = alpha /
 * (alpha + beta))} with plain box bounds, so no general inequality constraint
 * is needed. Asymptotic standard errors come from the inverse of the
 * finite-difference Hessian of the log-likelihood at the optimum, in the
 * natural {@code (omega, alpha, beta)} coordinates.
 */
public record Garch11(double omega, double alpha, double beta) {

  public Garch11 {
    if (!(omega > 0.0) || !(alpha >= 0.0) || !(beta >= 0.0)) {
      throw new IllegalArgumentException("need omega>0, alpha>=0, beta>=0");
    }
  }

  /** Unconditional (long-run) variance {@code omega / (1 - alpha - beta)}. */
  public double longRunVariance() {
    double persistence = alpha + beta;
    if (!(persistence < 1.0)) {
      return Double.POSITIVE_INFINITY;
    }
    return omega / (1.0 - persistence);
  }

  /**
   * The conditional-variance path, seeded with the long-run variance.
   * {@code variance[t]} is the estimate before {@code returns[t]} is seen.
   */
  public double[] conditionalVariance(double[] returns) {
    double[] v = new double[returns.length];
    double s2 = Double.isFinite(longRunVariance()) ? longRunVariance() : Ewma.sampleVariance(returns);
    for (int t = 0; t < returns.length; t++) {
      v[t] = s2;
      s2 = omega + alpha * returns[t] * returns[t] + beta * s2;
    }
    return v;
  }

  /** Gaussian negative log-likelihood (up to a constant), plain {@code double}. */
  public static double negLogLikelihood(double omega, double alpha, double beta, double[] returns, double var0) {
    double s2 = var0;
    double nll = 0.0;
    for (double r : returns) {
      if (!(s2 > 0.0)) {
        return Double.POSITIVE_INFINITY;
      }
      nll += Math.log(s2) + r * r / s2;
      s2 = omega + alpha * r * r + beta * s2;
    }
    return 0.5 * nll;
  }

  /**
   * Fit {@code (omega, alpha, beta)} to {@code returns} (assumed zero-mean) by
   * maximum likelihood.
   */
  public static Fit fit(double[] returns) {
    if (returns.length < 20) {
      throw new IllegalArgumentException("need at least 20 observations, got " + returns.length);
    }
    double var0 = Ewma.sampleVariance(returns);

    // Fit in coordinates that are all O(1) and comparably scaled, so the
    // adjoint gradient is well conditioned for L-BFGS:
    //   omega        = omegaFrac * var0        (omegaFrac ~ 0.01 .. 0.2)
    //   persistence  = alpha + beta            (enforces alpha + beta < 1)
    //   share        = alpha / (alpha + beta)  (enforces alpha, beta >= 0)
    Calibrator.Result res = Calibrator.of(rec -> {
      SDouble omega = rec.input("omegaFrac", 0.05).mul(var0);
      SDouble persistence = rec.input("persistence", 0.95);
      SDouble share = rec.input("share", 0.10);
      SDouble alpha = persistence.mul(share);
      SDouble beta = persistence.sub(alpha);

      SDouble s2 = rec.constant(var0);
      SDouble nll = rec.constant(0.0);
      for (double ret : returns) {
        SDouble r2 = rec.constant(ret * ret);
        nll = nll.add(s2.log()).add(r2.div(s2));
        s2 = omega.add(alpha.mul(r2)).add(beta.mul(s2));
      }
      rec.output(nll.mul(0.5));
    })
        .parameter("omegaFrac", 0.05, 1e-6, 10.0)
        .parameter("persistence", 0.95, 1e-6, 0.99999)
        .parameter("share", 0.10, 0.0, 1.0)
        .maxIterations(400)
        .tolerance(1e-9)
        // The likelihood unrolls one node per observation, so the tape has tens
        // of thousands of nodes — past what the straight-line bytecode kernel
        // can emit. The scalar engine replays a tape of any size.
        .on("cpu")
        .solve();

    double omega = res.parameters().get("omegaFrac") * var0;
    double persistence = res.parameters().get("persistence");
    double share = res.parameters().get("share");
    double alpha = persistence * share;
    double beta = persistence - alpha;

    Garch11 params = new Garch11(Math.max(omega, 1e-12), Math.max(alpha, 0.0), Math.max(beta, 0.0));
    double logLik = -negLogLikelihood(params.omega(), params.alpha(), params.beta(), returns, var0);
    double[] se = standardErrors(params, returns, var0);
    return new Fit(params, logLik, res.iterations(), res.converged(), se, params.conditionalVariance(returns));
  }

  private static double[] standardErrors(Garch11 p, double[] returns, double var0) {
    double[] theta = {p.omega(), p.alpha(), p.beta()};
    double[] h = {Math.max(1e-6, 1e-3 * theta[0]), 1e-4, 1e-4};
    double[][] hess = new double[3][3];
    for (int i = 0; i < 3; i++) {
      for (int j = i; j < 3; j++) {
        hess[i][j] = secondPartial(theta, i, j, h, returns, var0);
        hess[j][i] = hess[i][j];
      }
    }
    double[][] cov = invert3x3(hess);
    double[] se = new double[3];
    for (int i = 0; i < 3; i++) {
      se[i] = cov == null || !(cov[i][i] > 0.0) ? Double.NaN : Math.sqrt(cov[i][i]);
    }
    return se;
  }

  private static double secondPartial(double[] t, int i, int j, double[] h, double[] returns, double var0) {
    double[] pp = t.clone();
    double[] pm = t.clone();
    double[] mp = t.clone();
    double[] mm = t.clone();
    pp[i] += h[i]; pp[j] += h[j];
    pm[i] += h[i]; pm[j] -= h[j];
    mp[i] -= h[i]; mp[j] += h[j];
    mm[i] -= h[i]; mm[j] -= h[j];
    double f = 0.0;
    f += nll(pp, returns, var0);
    f -= nll(pm, returns, var0);
    f -= nll(mp, returns, var0);
    f += nll(mm, returns, var0);
    return f / (4.0 * h[i] * h[j]);
  }

  private static double nll(double[] t, double[] returns, double var0) {
    return negLogLikelihood(t[0], t[1], t[2], returns, var0);
  }

  private static double[][] invert3x3(double[][] a) {
    double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if (Math.abs(det) < 1e-300) {
      return null;
    }
    double inv = 1.0 / det;
    double[][] r = new double[3][3];
    r[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv;
    r[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv;
    r[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv;
    r[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv;
    r[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv;
    r[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv;
    r[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv;
    r[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv;
    r[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv;
    return r;
  }

  /** Named parameter map, for reporting. */
  public Map<String, Double> asMap() {
    Map<String, Double> m = new LinkedHashMap<>();
    m.put("omega", omega);
    m.put("alpha", alpha);
    m.put("beta", beta);
    return m;
  }
}
