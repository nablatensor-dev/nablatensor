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

import com.nablatensor.quant.OptionType;

/**
 * Merton's (1976) jump-diffusion price for a European option: a diffusion with
 * volatility {@code sigma} plus a compound-Poisson jump component of intensity
 * {@code lambda} whose multiplicative jump size is lognormal,
 * {@code ln Y ~ N(muJ, deltaJ^2)}.
 *
 * <p>The price is the Poisson-weighted average of Black-Scholes prices, one per
 * possible number of jumps to expiry:
 *
 * <pre>{@code
 * kappa   = exp(muJ + deltaJ^2 / 2) - 1            (expected relative jump size)
 * lambda' = lambda (1 + kappa)
 * sigma_n = sqrt(sigma^2 + n deltaJ^2 / T)
 * r_n     = r - lambda kappa + n ln(1 + kappa) / T
 * price   = sum_n  e^{-lambda' T} (lambda' T)^n / n!  *  BSM(S, K, T, r_n, sigma_n)
 * }</pre>
 *
 * <p>The series is truncated once the cumulative Poisson weight is within
 * {@code 1e-14} of one (and at {@code n = 256} as a hard stop). As
 * {@code lambda -> 0} only the {@code n = 0} term survives and the price
 * collapses to {@link GeneralizedBsm} with {@code q = 0}.
 *
 * <p>This is the analytic oracle the {@code MertonJumpModel} Monte-Carlo step
 * block (feature F7) is validated against.
 */
public final class MertonJumpDiffusion {

  private static final int MAX_TERMS = 256;
  private static final double WEIGHT_EPS = 1.0e-14;

  private MertonJumpDiffusion() {
  }

  /**
   * @param type         call or put
   * @param spot         spot {@code S}
   * @param strike       strike {@code K}
   * @param maturity     time to expiry in years {@code T}
   * @param rate         continuously-compounded risk-free rate {@code r}
   * @param vol          diffusion volatility {@code sigma}
   * @param jumpIntensity Poisson intensity {@code lambda} (expected jumps per year)
   * @param jumpMean     mean of the log jump size {@code muJ}
   * @param jumpVol      standard deviation of the log jump size {@code deltaJ}
   */
  public static AnalyticGreeks of(OptionType type, double spot, double strike, double maturity,
                                  double rate, double vol, double jumpIntensity,
                                  double jumpMean, double jumpVol) {
    if (maturity <= 0.0 || vol <= 0.0) {
      return AnalyticGreeks.intrinsic(
          CostOfCarry.price(type, spot, strike, maturity, rate, rate, vol));
    }
    Greeking.Price5 f = (s, k, t, r, v) ->
        price(type, s, k, t, r, v, jumpIntensity, jumpMean, jumpVol);
    return Greeking.central(f, spot, strike, maturity, rate, vol);
  }

  /** Bare price — the truncated Poisson series. */
  public static double price(OptionType type, double spot, double strike, double maturity,
                             double rate, double vol, double jumpIntensity,
                             double jumpMean, double jumpVol) {
    if (maturity <= 0.0 || vol <= 0.0) {
      return CostOfCarry.price(type, spot, strike, maturity, rate, rate, vol);
    }
    double kappa = Math.exp(jumpMean + 0.5 * jumpVol * jumpVol) - 1.0;
    double lambdaPrime = jumpIntensity * (1.0 + kappa);
    double lambdaT = lambdaPrime * maturity;

    double sum = 0.0;
    double weight = Math.exp(-lambdaT); // n = 0
    double cumulative = 0.0;
    for (int n = 0; n < MAX_TERMS; n++) {
      double sigmaN = Math.sqrt(vol * vol + n * jumpVol * jumpVol / maturity);
      double rN = rate - jumpIntensity * kappa + n * Math.log1p(kappa) / maturity;
      // b = r_n : the per-term diffusion carries the full drift, no separate yield.
      sum += weight * CostOfCarry.price(type, spot, strike, maturity, rN, rN, sigmaN);
      cumulative += weight;
      if (cumulative > 1.0 - WEIGHT_EPS && n > 0) {
        break;
      }
      weight *= lambdaT / (n + 1);
    }
    return sum;
  }
}
