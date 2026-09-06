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
 * Turns a five-argument closed-form price {@code f(S, K, T, r, sigma)} into an
 * {@link AnalyticGreeks} by central differencing.
 *
 * <p>The pricers in this package are exact functions evaluated in {@code double},
 * so the only error in the differenced sensitivities is the {@code O(h^2)}
 * truncation of the stencil (and, for {@link #gamma}, {@code O(eps/h^2)}
 * round-off). With the step sizes chosen here that is at the {@code 1e-7} level
 * for a textbook at-the-money one-year option — well inside what a Monte-Carlo
 * adjoint check needs, and with no Monte-Carlo noise of its own.
 *
 * <p>Writing the six Greeks this way rather than transcribing six more closed
 * forms per model keeps the oracle small and removes a class of copy error; the
 * package tests still pin the differenced values against published Black-Scholes
 * Greeks.
 */
final class Greeking {

  /** A closed-form price as a function of the five market inputs. */
  @FunctionalInterface
  interface Price5 {
    double at(double spot, double strike, double maturity, double rate, double vol);
  }

  private Greeking() {
  }

  /** First-derivative relative step. */
  private static final double H1 = 1.0e-5;
  /** Second-derivative (gamma) relative step — larger, to keep {@code eps/h^2} in check. */
  private static final double H2 = 1.0e-3;

  static AnalyticGreeks central(Price5 f, double s, double k, double t, double r, double sigma) {
    double price = f.at(s, k, t, r, sigma);

    double hs = H1 * Math.max(1.0, Math.abs(s));
    double hk = H1 * Math.max(1.0, Math.abs(k));
    double ht = H1 * Math.max(1.0, Math.abs(t));
    double hr = H1 * Math.max(1.0, Math.abs(r));
    double hv = H1 * Math.max(1.0, Math.abs(sigma));
    double hg = H2 * Math.max(1.0, Math.abs(s));

    double delta = (f.at(s + hs, k, t, r, sigma) - f.at(s - hs, k, t, r, sigma)) / (2.0 * hs);
    double gamma = (f.at(s + hg, k, t, r, sigma) - 2.0 * price + f.at(s - hg, k, t, r, sigma)) / (hg * hg);
    double vega = (f.at(s, k, t, r, sigma + hv) - f.at(s, k, t, r, sigma - hv)) / (2.0 * hv);
    double theta = (f.at(s, k, t + ht, r, sigma) - f.at(s, k, t - ht, r, sigma)) / (2.0 * ht);
    double rho = (f.at(s, k, t, r + hr, sigma) - f.at(s, k, t, r - hr, sigma)) / (2.0 * hr);
    double dvdk = (f.at(s, k + hk, t, r, sigma) - f.at(s, k - hk, t, r, sigma)) / (2.0 * hk);

    return new AnalyticGreeks(price, delta, gamma, vega, theta, rho, dvdk);
  }
}
