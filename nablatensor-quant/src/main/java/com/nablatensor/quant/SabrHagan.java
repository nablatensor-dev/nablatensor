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
import com.nablatensor.ops.SpecialFn;

/**
 * Hagan et al. (2002) lognormal-implied-volatility approximation for SABR.
 *
 * <p>Two forms: a plain {@code double} {@link #blackVol(double, double, double,
 * double, double, double, double)} for generating reference quotes, and a
 * tape-level {@link #blackVol(AadRecorder, SDouble, SDouble, SDouble, SDouble,
 * double, double, double)} whose {@code alpha}, {@code beta}, {@code rho} and
 * {@code nu} are differentiable — the model function inside {@link Calibrator}.
 */
public final class SabrHagan {

  private SabrHagan() {
  }

  /** Reference (host) Black vol. */
  public static double blackVol(double alpha, double beta, double rho, double nu,
                                double forward, double strike, double maturity) {
    double oneMinusBeta = 1.0 - beta;
    double logFK = Math.log(forward / strike);
    double fkBeta = Math.pow(forward * strike, oneMinusBeta / 2.0);

    double aDen = fkBeta * (1.0
        + oneMinusBeta * oneMinusBeta / 24.0 * logFK * logFK
        + Math.pow(oneMinusBeta, 4) / 1920.0 * Math.pow(logFK, 4));
    double a = alpha / aDen;

    double b = 1.0 + maturity * (
        oneMinusBeta * oneMinusBeta / 24.0 * alpha * alpha / (fkBeta * fkBeta)
        + 0.25 * rho * beta * nu * alpha / fkBeta
        + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu);

    if (Math.abs(logFK) < 1e-12) {
      return a * b;
    }
    double z = nu / alpha * fkBeta * logFK;
    double xz = Math.log((Math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho));
    return a * (z / xz) * b;
  }

  /**
   * Tape-level Black vol. {@code forward}, {@code strike} and {@code maturity}
   * are constants; the four SABR parameters are differentiable. Requires
   * {@code strike != forward} (use a small offset for the ATM point).
   */
  public static SDouble blackVol(AadRecorder rec, SDouble alpha, SDouble beta, SDouble rho, SDouble nu,
                                 double forward, double strike, double maturity) {
    if (Math.abs(forward - strike) < 1e-12) {
      throw new IllegalArgumentException("tape-level blackVol needs strike != forward");
    }
    double logFK = Math.log(forward / strike);
    SDouble oneMinusBeta = beta.neg().add(1.0);

    // fkBeta = (F K) ^ ((1-beta)/2)
    SDouble fkBeta = SpecialFn.pow(rec.constant(forward * strike), oneMinusBeta.mul(0.5));

    SDouble aDen = fkBeta.mul(oneMinusBeta.mul(oneMinusBeta).mul(logFK * logFK / 24.0)
        .add(SpecialFn.pow(oneMinusBeta, 4.0).mul(Math.pow(logFK, 4) / 1920.0))
        .add(1.0));
    SDouble a = alpha.div(aDen);

    SDouble b = oneMinusBeta.mul(oneMinusBeta).mul(alpha.mul(alpha).div(fkBeta.mul(fkBeta))).mul(1.0 / 24.0)
        .add(rho.mul(beta).mul(nu).mul(alpha).div(fkBeta).mul(0.25))
        .add(rho.mul(rho).mul(-3.0).add(2.0).mul(nu.mul(nu)).mul(1.0 / 24.0))
        .mul(maturity)
        .add(1.0);

    SDouble z = nu.div(alpha).mul(fkBeta).mul(logFK);
    SDouble root = z.mul(z).sub(rho.mul(z).mul(2.0)).add(1.0).sqrt();
    SDouble xz = root.add(z).sub(rho).div(rho.neg().add(1.0)).log();
    return a.mul(z.div(xz)).mul(b);
  }
}
