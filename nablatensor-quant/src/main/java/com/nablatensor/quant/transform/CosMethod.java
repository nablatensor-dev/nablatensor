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

import com.nablatensor.quant.OptionType;

/**
 * The Fang-Oosterlee COS method: a European option price as a cosine series in
 * the risk-neutral density of the log-return, reconstructed from a
 * {@link CharacteristicFunction}. Spectral convergence — for a smooth density a
 * few hundred terms give machine-precision accuracy, and a whole strike slice
 * costs one pass of the same {@code phi} evaluations.
 *
 * <pre>{@code
 * price = e^{-rT} * sum_{k=0}^{N-1}' Re{ phi(k pi / (b-a)) e^{-i k pi a / (b-a)} } * U_k
 * }</pre>
 *
 * where {@code [a, b]} is the truncation range from the cumulants and {@code U_k}
 * are the payoff cosine coefficients. Puts are taken from the call by parity.
 */
public final class CosMethod {

  private static final int DEFAULT_TERMS = 256;
  private static final double DEFAULT_L = 10.0;

  private CosMethod() {
  }

  public static double price(CharacteristicFunction cf, OptionType type,
                             double spot, double strike, double rate, double maturity) {
    return price(cf, type, spot, strike, rate, maturity, DEFAULT_TERMS, DEFAULT_L);
  }

  public static double price(CharacteristicFunction cf, OptionType type,
                             double spot, double strike, double rate, double maturity,
                             int terms, double rangeWidths) {
    double c1 = cf.cumulant1(maturity);
    double c2 = cf.cumulant2(maturity);
    double c4 = cf.cumulant4(maturity);
    double halfWidth = rangeWidths * Math.sqrt(Math.abs(c2) + Math.sqrt(Math.abs(c4)));
    double a = c1 - halfWidth;
    double b = c1 + halfWidth;

    double xStar = Math.log(strike / spot);                    // log-moneyness; payoff kink of X_T
    double bma = b - a;
    double call = 0.0;
    for (int k = 0; k < terms; k++) {
      double omega = k * Math.PI / bma;
      Complex unit = cf.phi(omega, maturity).mul(new Complex(0.0, -omega * a).exp());
      double uk = callCoefficient(k, xStar, a, b, spot, strike);
      double term = unit.re() * uk;
      call += (k == 0) ? 0.5 * term : term;
    }
    call *= Math.exp(-rate * maturity);
    call = Math.max(call, 0.0);

    if (type == OptionType.CALL) {
      return call;
    }
    return call - spot + strike * Math.exp(-rate * maturity);   // put-call parity
  }

  /**
   * {@code U_k} for a call payoff {@code (S_0 e^y - K)^+} on {@code [xStar, b]}:
   * {@code (2/(b-a)) (S_0 chi_k(xStar, b) - K psi_k(xStar, b))}.
   */
  private static double callCoefficient(int k, double xStar, double a, double b,
                                        double spot, double strike) {
    double bma = b - a;
    double chi = chi(k, xStar, b, a, bma);
    double psi = psi(k, xStar, b, a, bma);
    return 2.0 / bma * (spot * chi - strike * psi);
  }

  private static double chi(int k, double c, double d, double a, double bma) {
    double kpi = k * Math.PI / bma;
    double cosD = Math.cos(kpi * (d - a));
    double cosC = Math.cos(kpi * (c - a));
    double sinD = Math.sin(kpi * (d - a));
    double sinC = Math.sin(kpi * (c - a));
    double ed = Math.exp(d);
    double ec = Math.exp(c);
    return (cosD * ed - cosC * ec + kpi * (sinD * ed - sinC * ec)) / (1.0 + kpi * kpi);
  }

  private static double psi(int k, double c, double d, double a, double bma) {
    if (k == 0) {
      return d - c;
    }
    double kpi = k * Math.PI / bma;
    return (Math.sin(kpi * (d - a)) - Math.sin(kpi * (c - a))) / kpi;
  }
}
