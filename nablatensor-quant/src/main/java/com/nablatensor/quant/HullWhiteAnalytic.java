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

import com.nablatensor.quant.analytic.Normal;

/**
 * The Hull-White one-factor short-rate model made term-structure-consistent:
 * given today's discount curve {@code P^M(0, .)} and the two parameters
 * {@code (a, sigma)}, it reprices that curve exactly and prices European bond
 * options, caplets/floorlets and — by Jamshidian decomposition — European
 * swaptions in closed form.
 *
 * <p>This is the analytic companion to the {@link HullWhite1F} Monte-Carlo step
 * block (whose flat-forward assumption it removes) and the pricing engine
 * {@link HullWhiteCalibration} fits {@code (a, sigma)} to a swaption grid with.
 *
 * <p>The bond reconstitution is
 * <pre>{@code
 * B(t,T) = (1 - e^{-a(T-t)}) / a
 * A(t,T) = (P^M(0,T)/P^M(0,t)) exp( B(t,T) f^M(0,t) - (sigma^2/(4a))(1 - e^{-2at}) B(t,T)^2 )
 * P(t,T | r_t) = A(t,T) e^{-B(t,T) r_t}
 * }</pre>
 * so today's curve is fitted with no separate {@code theta(t)} calibration;
 * {@link #theta} is exposed for a simulation that needs it. The {@code a -> 0}
 * limit is the Ho-Lee model and is handled without dividing by {@code a}.
 */
public final class HullWhiteAnalytic {

  private static final double SMALL_A = 1.0e-8;

  private final YieldCurve curve;
  private final double a;
  private final double sigma;
  private final double r0;

  private HullWhiteAnalytic(YieldCurve curve, double a, double sigma) {
    if (!(a >= 0.0) || !(sigma >= 0.0)) {
      throw new IllegalArgumentException("need a >= 0 and sigma >= 0");
    }
    this.curve = curve;
    this.a = a;
    this.sigma = sigma;
    this.r0 = instantaneousForward(0.0);
  }

  public static HullWhiteAnalytic of(YieldCurve discountCurve, double a, double sigma) {
    return new HullWhiteAnalytic(discountCurve, a, sigma);
  }

  public static HullWhiteAnalytic of(CurveSet curves, double a, double sigma) {
    return new HullWhiteAnalytic(curves.discount(), a, sigma);
  }

  public double a() {
    return a;
  }

  public double sigma() {
    return sigma;
  }

  /** Today's initial short rate {@code r(0) = f^M(0, 0)}. */
  public double r0() {
    return r0;
  }

  /** {@code B(t, T) = (1 - e^{-a(T-t)}) / a}, with the {@code a -> 0} limit {@code T - t}. */
  public double bFactor(double t, double bondMaturity) {
    double dtau = bondMaturity - t;
    return a < SMALL_A ? dtau : -Math.expm1(-a * dtau) / a;
  }

  /** {@code (1 - e^{-2 a t}) / (2 a)}, with the {@code a -> 0} limit {@code t}. */
  private double variancePrefactor(double t) {
    return a < SMALL_A ? t : -Math.expm1(-2.0 * a * t) / (2.0 * a);
  }

  private double pm(double t) {
    return t <= 0.0 ? 1.0 : curve.discountFactor(t);
  }

  /** Instantaneous forward {@code f^M(0, t) = -d ln P^M / dt}. */
  public double instantaneousForward(double t) {
    double e = 1.0e-5;
    double tt = Math.max(t, e);
    return -(Math.log(pm(tt + e)) - Math.log(pm(tt - e))) / (2.0 * e);
  }

  /**
   * The Hull-White drift term {@code theta(t) = d f^M/dt + a f^M(0,t) +
   * (sigma^2 / (2a)) (1 - e^{-2 a t})} — needed only to simulate the short rate;
   * the analytic prices in this class do not use it.
   */
  public double theta(double t) {
    double e = 1.0e-4;
    double dfdt = (instantaneousForward(t + e) - instantaneousForward(Math.max(t - e, 0.0)))
        / (t < e ? (t + e) : 2.0 * e);
    double lastTerm = a < SMALL_A
        ? sigma * sigma * t
        : sigma * sigma / (2.0 * a) * (-Math.expm1(-2.0 * a * t));
    return dfdt + a * instantaneousForward(t) + lastTerm;
  }

  /** Bond {@code P(t, T)} reconstituted from a realised short rate {@code r(t)}. */
  public double bondReconstitution(double t, double bondMaturity, double shortRate) {
    double b = bFactor(t, bondMaturity);
    // sigma^2/(4a) (1 - e^{-2at}) B^2, with the a->0 limit sigma^2 (t/2) B^2.
    double sigmaTerm = a < SMALL_A
        ? 0.5 * sigma * sigma * t * b * b
        : sigma * sigma / (4.0 * a) * (-Math.expm1(-2.0 * a * t)) * b * b;
    double lnA = Math.log(pm(bondMaturity) / pm(t)) + b * instantaneousForward(t) - sigmaTerm;
    return Math.exp(lnA - b * shortRate);
  }

  // ---- options on a zero-coupon bond -----------------------------------

  private double bondOptionVol(double optionExpiry, double bondMaturity) {
    return sigma * Math.sqrt(variancePrefactor(optionExpiry)) * bFactor(optionExpiry, bondMaturity);
  }

  /** European call on {@code P(T, S)} struck at {@code K}, {@code T < S}. */
  public double zeroBondCall(double optionExpiry, double bondMaturity, double strike) {
    double t = optionExpiry;
    double s = bondMaturity;
    double sp = bondOptionVol(t, s);
    if (sp <= 0.0) {
      return Math.max(pm(s) - strike * pm(t), 0.0);
    }
    double h = Math.log(pm(s) / (pm(t) * strike)) / sp + 0.5 * sp;
    return pm(s) * Normal.cdf(h) - strike * pm(t) * Normal.cdf(h - sp);
  }

  /** European put on {@code P(T, S)} struck at {@code K}, {@code T < S}. */
  public double zeroBondPut(double optionExpiry, double bondMaturity, double strike) {
    double t = optionExpiry;
    double s = bondMaturity;
    double sp = bondOptionVol(t, s);
    if (sp <= 0.0) {
      return Math.max(strike * pm(t) - pm(s), 0.0);
    }
    double h = Math.log(pm(s) / (pm(t) * strike)) / sp + 0.5 * sp;
    return strike * pm(t) * Normal.cdf(-h + sp) - pm(s) * Normal.cdf(-h);
  }

  // ---- caplet / floorlet ----------------------------------------------

  /**
   * A caplet on the simply-compounded rate for {@code [resetTime, resetTime +
   * accrual]}, strike rate {@code strikeRate}, unit notional — priced as
   * {@code (1 + K tau)} puts on {@code P(T, T+tau)} struck at {@code 1/(1 + K tau)}.
   */
  public double caplet(double resetTime, double accrual, double strikeRate) {
    double kBond = 1.0 / (1.0 + strikeRate * accrual);
    return (1.0 + strikeRate * accrual) * zeroBondPut(resetTime, resetTime + accrual, kBond);
  }

  /** A floorlet — {@code (1 + K tau)} calls on the same bond. */
  public double floorlet(double resetTime, double accrual, double strikeRate) {
    double kBond = 1.0 / (1.0 + strikeRate * accrual);
    return (1.0 + strikeRate * accrual) * zeroBondCall(resetTime, resetTime + accrual, kBond);
  }

  /** A cap: the sum of its caplets on an equally spaced schedule. */
  public double cap(double firstReset, double accrual, int periods, double strikeRate) {
    double v = 0.0;
    for (int i = 0; i < periods; i++) {
      v += caplet(firstReset + i * accrual, accrual, strikeRate);
    }
    return v;
  }

  // ---- European swaption (Jamshidian) --------------------------------

  /**
   * Physically-settled European payer swaption: expiry {@code expiry}, then
   * {@code periods} fixed payments of accrual {@code accrual} at
   * {@code expiry + i*accrual}, strike {@code strikeRate}, unit notional.
   *
   * <p>Jamshidian: the payer swaption is a put on the fixed-coupon bond, which
   * decomposes into a portfolio of puts on the individual zero-coupon bonds once
   * the critical short rate {@code r*} (where the coupon bond is worth par) is
   * found by a 1-D solve.
   */
  public double payerSwaption(double expiry, double accrual, int periods, double strikeRate) {
    return swaption(expiry, accrual, periods, strikeRate, true);
  }

  public double receiverSwaption(double expiry, double accrual, int periods, double strikeRate) {
    return swaption(expiry, accrual, periods, strikeRate, false);
  }

  private double swaption(double expiry, double accrual, int periods, double strikeRate, boolean payer) {
    double[] payTimes = new double[periods];
    double[] coupon = new double[periods];
    for (int i = 0; i < periods; i++) {
      payTimes[i] = expiry + (i + 1) * accrual;
      coupon[i] = strikeRate * accrual + (i == periods - 1 ? 1.0 : 0.0);
    }
    double rStar = solveCriticalRate(expiry, payTimes, coupon);

    double value = 0.0;
    for (int i = 0; i < periods; i++) {
      double ki = bondReconstitution(expiry, payTimes[i], rStar);
      value += coupon[i] * (payer
          ? zeroBondPut(expiry, payTimes[i], ki)
          : zeroBondCall(expiry, payTimes[i], ki));
    }
    return value;
  }

  /** {@code r*} such that {@code sum_i coupon_i P(expiry, payTimes_i | r*) = 1}. */
  private double solveCriticalRate(double expiry, double[] payTimes, double[] coupon) {
    java.util.function.DoubleUnaryOperator couponBond = r -> {
      double v = 0.0;
      for (int i = 0; i < payTimes.length; i++) {
        v += coupon[i] * bondReconstitution(expiry, payTimes[i], r);
      }
      return v - 1.0;
    };
    double lo = -0.50;
    double hi = 1.00;
    double flo = couponBond.applyAsDouble(lo);
    double fhi = couponBond.applyAsDouble(hi);
    // couponBond is strictly decreasing; widen if the bracket is wrong.
    for (int k = 0; k < 40 && flo * fhi > 0.0; k++) {
      lo -= 0.5;
      hi += 1.0;
      flo = couponBond.applyAsDouble(lo);
      fhi = couponBond.applyAsDouble(hi);
    }
    for (int k = 0; k < 200; k++) {
      double mid = 0.5 * (lo + hi);
      double fm = couponBond.applyAsDouble(mid);
      if (Math.abs(fm) < 1e-14 || (hi - lo) < 1e-14) {
        return mid;
      }
      if (flo * fm <= 0.0) {
        hi = mid;
      } else {
        lo = mid;
        flo = fm;
      }
    }
    return 0.5 * (lo + hi);
  }
}
