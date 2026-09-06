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
package com.nablatensor.credit;

import com.nablatensor.quant.analytic.Normal;

/**
 * The one-factor Gaussian copula: each name's latent variable is
 * {@code X_i = sqrt(rho) M + sqrt(1 - rho) Z_i} with a common systemic factor
 * {@code M} and idiosyncratic {@code Z_i}, all standard normal. Name {@code i}
 * defaults by {@code t} when {@code X_i < Phi^{-1}(PD_i(t))}.
 *
 * <p>Conditional on {@code M}, defaults are independent with probability
 * {@code p_i(M) = Phi( (Phi^{-1}(PD_i) - sqrt(rho) M) / sqrt(1 - rho) )} — the
 * fact the {@link PortfolioLossDistribution} recursion and the
 * large-homogeneous-pool (Vasicek) limit both rest on.
 */
public final class OneFactorGaussianCopula {

  private OneFactorGaussianCopula() {
  }

  /** Default probability of a name conditional on the systemic factor {@code m}. */
  public static double conditionalDefaultProbability(double unconditionalPd, double rho, double m) {
    if (rho <= 0.0) {
      return unconditionalPd;
    }
    if (rho >= 1.0) {
      return m < Normal.inverseCdf(unconditionalPd) ? 1.0 : 0.0;
    }
    double threshold = Normal.inverseCdf(unconditionalPd);
    return Normal.cdf((threshold - Math.sqrt(rho) * m) / Math.sqrt(1.0 - rho));
  }

  /**
   * The Vasicek large-homogeneous-pool loss distribution: the probability that
   * the fractional portfolio loss is at most {@code x}, for unconditional
   * default probability {@code pd}, correlation {@code rho} and loss given
   * default {@code lgd}.
   */
  public static double vasicekLossCdf(double x, double pd, double rho, double lgd) {
    double lossFraction = x / lgd;
    if (lossFraction >= 1.0) {
      return 1.0;
    }
    if (lossFraction <= 0.0) {
      return 0.0;
    }
    // P(conditional default prob <= lossFraction) as a function of M.
    double k = Normal.inverseCdf(lossFraction);
    double num = Math.sqrt(1.0 - rho) * k - Normal.inverseCdf(pd);
    return Normal.cdf(num / Math.sqrt(rho));
  }

  /** Expected fractional portfolio loss in the Vasicek limit ({@code = pd * lgd}). */
  public static double vasicekExpectedLoss(double pd, double lgd) {
    return pd * lgd;
  }
}
