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
package com.nablatensor.quant.adjust;

/**
 * Closed-form convexity adjustments — the corrections that turn a rate quoted or
 * observed in the "wrong" measure into its forward value.
 *
 * <ul>
 *   <li><b>Eurodollar futures</b>: the futures rate exceeds the forward rate
 *       because a futures contract is marked to market daily; the Hull-White
 *       one-factor adjustment reduces to {@code sigma^2 t1 (t1 + 2 tau) / 2} as
 *       {@code a -> 0} (the rigorous futures-rate convexity — Hull's textbook
 *       {@code sigma^2 t1 t2 / 2} drops one sub-term).</li>
 *   <li><b>LIBOR in arrears</b>: a rate that pays on its own fixing date rather
 *       than one accrual period later is worth more than its forward, by
 *       {@code tau L0^2 (e^{sigma^2 T} - 1) / (1 + tau L0)} (exact for a
 *       lognormal forward).</li>
 *   <li><b>CMS</b>: a swap rate paid once rather than as an annuity carries a
 *       convexity adjustment {@code -0.5 y0^2 sigma^2 T G''(y0)/G'(y0)}, with
 *       {@code G} the flat-yield annuity function.</li>
 * </ul>
 */
public final class ConvexityAdjustment {

  private static final double SMALL_A = 1.0e-8;

  private ConvexityAdjustment() {
  }

  /** {@code B(u, v) = (1 - e^{-a(v-u)}) / a}, with the {@code a -> 0} limit {@code v - u}. */
  private static double b(double a, double u, double v) {
    double d = v - u;
    return a < SMALL_A ? d : -Math.expm1(-a * d) / a;
  }

  /** The Eurodollar-futures convexity adjustment and its Hull-White parameter sensitivities. */
  public record FuturesConvexity(double rateAdjustment, double dSigma, double dMeanReversion) {
    /** {@code forwardRate = futuresRate - rateAdjustment}. */
    public double forwardFromFutures(double futuresRate) {
      return futuresRate - rateAdjustment;
    }
  }

  /**
   * Hull-White one-factor convexity adjustment for the rate covering
   * {@code [t1, t2]} implied by a Eurodollar future expiring at {@code t1}:
   *
   * <pre>{@code
   * CA = (1/(t2-t1)) * (sigma^2/(2a)) * B(t1,t2) * [ B(t1,t2)(1 - e^{-2 a t1}) + a B(0,t1)^2 ]
   * }</pre>
   *
   * the rigorous {@code E^Q[(1/P(t1,t2) - 1)/tau] - forwardRate} to first order,
   * which tends to {@code sigma^2 t1 (t1 + 2 tau) / 2} as {@code a -> 0}. (The
   * {@code 1/P(0,t2)} scaling — a fraction of a percent — is dropped to keep the
   * signature curve-free.)
   *
   * @param meanReversion Hull-White {@code a} ({@code >= 0}; near-zero gives Ho-Lee)
   * @param sigma         absolute (normal) short-rate volatility
   * @param t1            futures expiry / rate start, years
   * @param t2            rate end, years
   */
  public static FuturesConvexity eurodollarFutures(double meanReversion, double sigma,
                                                   double t1, double t2) {
    double ca = eurodollarValue(meanReversion, sigma, t1, t2);
    double hS = 1e-6 * Math.max(1.0, sigma);
    double hA = 1e-6 * Math.max(1.0, meanReversion);
    double dSigma = (eurodollarValue(meanReversion, sigma + hS, t1, t2)
        - eurodollarValue(meanReversion, sigma - hS, t1, t2)) / (2 * hS);
    double dA = (eurodollarValue(meanReversion + hA, sigma, t1, t2)
        - eurodollarValue(Math.max(meanReversion - hA, 0.0), sigma, t1, t2)) / (2 * hA);
    return new FuturesConvexity(ca, dSigma, dA);
  }

  private static double eurodollarValue(double a, double sigma, double t1, double t2) {
    double tau = t2 - t1;
    if (a < SMALL_A) {
      return 0.5 * sigma * sigma * t1 * (t1 + 2.0 * tau);
    }
    double b12 = b(a, t1, t2);
    double b01 = b(a, 0.0, t1);
    return (1.0 / tau) * (sigma * sigma / (2.0 * a)) * b12
        * (b12 * (-Math.expm1(-2.0 * a * t1)) + a * b01 * b01);
  }

  /**
   * LIBOR-in-arrears adjustment: a forward rate {@code forward} that fixes at
   * {@code fixingTime} and pays then (rather than at {@code fixingTime + accrual})
   * has expected value {@code forward + L0^2 sigma^2 tau T / (1 + tau L0)} under
   * the pay-date measure.
   *
   * @param forward    the forward rate {@code L0}
   * @param blackVol   the lognormal (Black) volatility of the forward rate
   * @param accrual    the accrual fraction {@code tau}
   * @param fixingTime the fixing time {@code T} in years
   */
  public static Adjustment inArrears(double forward, double blackVol, double accrual, double fixingTime) {
    double adj = accrual * forward * forward * Math.expm1(blackVol * blackVol * fixingTime)
        / (1.0 + accrual * forward);
    return new Adjustment(forward, adj);
  }

  /**
   * CMS convexity adjustment (the convexity term only) for a swap rate observed
   * at {@code expiry} and paid once:
   * {@code -0.5 y0^2 sigma^2 T G''(y0)/G'(y0)}, with the flat-yield annuity
   * {@code G(y) = sum_{i=1}^{n} (1/m) / (1 + y/m)^i}.
   *
   * @param forwardSwapRate  {@code y0}
   * @param blackVol         lognormal vol of the swap rate
   * @param expiry           observation time {@code T} in years
   * @param paymentsPerYear  {@code m} (e.g. 2 for semi-annual)
   * @param numberOfPayments {@code n} (tenor in years times {@code m})
   */
  public static Adjustment cms(double forwardSwapRate, double blackVol, double expiry,
                               int paymentsPerYear, int numberOfPayments) {
    double y = forwardSwapRate;
    double h = 1e-5 * Math.max(1.0, y);
    double g = annuityFunction(y, paymentsPerYear, numberOfPayments);
    double gp = (annuityFunction(y + h, paymentsPerYear, numberOfPayments)
        - annuityFunction(y - h, paymentsPerYear, numberOfPayments)) / (2 * h);
    double gpp = (annuityFunction(y + h, paymentsPerYear, numberOfPayments)
        - 2 * g + annuityFunction(y - h, paymentsPerYear, numberOfPayments)) / (h * h);
    double adj = -0.5 * y * y * blackVol * blackVol * expiry * (gpp / gp);
    return new Adjustment(y, adj);
  }

  private static double annuityFunction(double y, int m, int n) {
    double s = 0.0;
    double base = 1.0 + y / m;
    double pow = base;
    for (int i = 1; i <= n; i++) {
      s += (1.0 / m) / pow;
      pow *= base;
    }
    return s;
  }
}
