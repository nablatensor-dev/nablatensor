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

/**
 * Variance-Gamma characteristic function of the log-return. VG is Brownian
 * motion with drift {@code theta} and volatility {@code sigma} evaluated at a
 * Gamma-distributed business time with unit mean rate and variance {@code nu}:
 *
 * <pre>{@code
 * phi(u) = exp( i u (r + omega) T ) (1 - i u theta nu + sigma^2 nu u^2 / 2)^{-T/nu}
 * omega  = (1/nu) ln(1 - theta nu - sigma^2 nu / 2)     (martingale correction)
 * }</pre>
 *
 * <p>As {@code nu -> 0} the Gamma time concentrates and VG -> Black-Scholes with
 * volatility {@code sigma}. This is also the closed-form pricing route for the
 * VG model that the F7 Monte-Carlo step block deferred.
 */
public record VarianceGammaCf(double rate, double sigma, double nu, double theta)
    implements CharacteristicFunction {

  private double omega() {
    return Math.log(1.0 - theta * nu - 0.5 * sigma * sigma * nu) / nu;
  }

  @Override
  public Complex phi(double u, double t) {
    Complex drift = new Complex(0.0, u * (rate + omega()) * t);
    // base = 1 - i u theta nu + sigma^2 nu u^2 / 2
    Complex base = new Complex(1.0 + 0.5 * sigma * sigma * nu * u * u, -u * theta * nu);
    return drift.exp().mul(base.pow(-t / nu));
  }

  @Override
  public double cumulant1(double t) {
    return (rate + omega() + theta) * t;
  }

  @Override
  public double cumulant2(double t) {
    return (sigma * sigma + nu * theta * theta) * t;
  }

  @Override
  public double cumulant4(double t) {
    return 3.0 * (sigma * sigma * sigma * sigma * nu
        + 2.0 * theta * theta * theta * theta * nu * nu * nu
        + 4.0 * sigma * sigma * theta * theta * nu * nu) * t;
  }
}
