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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.quant.adjust.Adjustment;
import com.nablatensor.quant.adjust.ConvexityAdjustment;
import com.nablatensor.quant.adjust.QuantoAdjustment;
import com.nablatensor.quant.adjust.TimingAdjustment;
import com.nablatensor.quant.analytic.AnalyticGreeks;
import com.nablatensor.quant.analytic.GeneralizedBsm;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Feature F8: each closed-form adjustment matches a Monte-Carlo of the same
 * cash flow in the "wrong" measure, and the Ho-Lee limit of the Eurodollar
 * futures adjustment is {@code sigma^2 t1 t2 / 2}.
 */
class AdjustmentTest {

  @Test
  void eurodollarFuturesHoLeeLimit() {
    double sigma = 0.011;
    double t1 = 2.0;
    double tau = 0.5;
    double t2 = t1 + tau;
    ConvexityAdjustment.FuturesConvexity fc =
        ConvexityAdjustment.eurodollarFutures(1e-9, sigma, t1, t2);
    double expected = 0.5 * sigma * sigma * t1 * (t1 + 2.0 * tau);
    assertEquals(expected, fc.rateAdjustment(), 1e-4 * expected, "Ho-Lee limit sigma^2 t1 (t1 + 2 tau) / 2");
    assertTrue(fc.dSigma() > 0.0, "adjustment grows with vol");
  }

  @Test
  void eurodollarFuturesMatchesHullWhiteMonteCarlo() {
    double r0 = 0.03;
    double a = 0.05;
    double sigma = 0.010;
    double t1 = 2.0;
    double tau = 0.25;
    double t2 = t1 + tau;

    // Flat curve at r0.
    double[] pillars = new double[10];
    double[] zeros = new double[10];
    for (int i = 0; i < 10; i++) {
      pillars[i] = i + 1.0;
      zeros[i] = r0;
    }
    YieldCurve curve = new YieldCurve(pillars, zeros);
    HullWhiteAnalytic hw = HullWhiteAnalytic.of(curve, a, sigma);

    double forwardRate = (curve.discountFactor(t1) / curve.discountFactor(t2) - 1.0) / tau;

    // The futures rate is E^Q[ simple rate over [t1,t2] ]. x(t1) is exactly
    // Gaussian, so draw it directly (no Euler bias) with antithetic pairs.
    double v = sigma * sigma * (-Math.expm1(-2.0 * a * t1)) / (2.0 * a);   // Var[x(t1)]
    double sd = Math.sqrt(v);
    double phi = sigma * sigma / (2 * a * a) * Math.pow(1 - Math.exp(-a * t1), 2);
    double fwdShort = hw.instantaneousForward(t1);
    int pairs = 2_000_000;
    Random rng = new Random(20260906L);
    double sumRate = 0.0;
    for (int p = 0; p < pairs; p++) {
      double z = rng.nextGaussian();
      for (double zz : new double[] {z, -z}) {
        double rT1 = fwdShort + sd * zz + phi;
        double pT1T2 = hw.bondReconstitution(t1, t2, rT1);
        sumRate += (1.0 / pT1T2 - 1.0) / tau;
      }
    }
    double futuresRate = sumRate / (2.0 * pairs);
    double caMc = futuresRate - forwardRate;
    double caClosed = ConvexityAdjustment.eurodollarFutures(a, sigma, t1, t2).rateAdjustment();

    assertEquals(caMc, caClosed, 0.04 * caMc + 1e-7, "Eurodollar convexity: closed form vs HW MC");
  }

  @Test
  void inArrearsMatchesMeasureChangeMonteCarlo() {
    double l0 = 0.03;
    double vol = 0.20;
    double tau = 0.5;
    double fixingTime = 3.0;

    // L_T lognormal, martingale under the T+tau-forward measure; antithetic pairs.
    Random rng = new Random(7L);
    int pairs = 3_000_000;
    double drift = -0.5 * vol * vol * fixingTime;
    double diff = vol * Math.sqrt(fixingTime);
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < pairs; i++) {
      double z = rng.nextGaussian();
      for (double zz : new double[] {z, -z}) {
        double lt = l0 * Math.exp(drift + diff * zz);
        double w = 1.0 + tau * lt;               // Radon-Nikodym to the pay-at-fixing measure
        num += lt * w;
        den += w;
      }
    }
    double adjMc = num / den - l0;
    double adjClosed = ConvexityAdjustment.inArrears(l0, vol, tau, fixingTime).adjustment();
    // The measure-change expectation is exactly tau L0^2 (e^{sigma^2 T} - 1) / (1 + tau L0).
    double adjExact = tau * l0 * l0 * Math.expm1(vol * vol * fixingTime) / (1.0 + tau * l0);

    assertTrue(adjClosed > 0.0, "in-arrears fixing is worth more than the forward");
    assertEquals(adjExact, adjClosed, 1e-12, "closed form is the exact lognormal value");
    // MC of a ratio-of-means estimator on a ~6e-5 convexity term: sanity, not precision.
    assertEquals(adjMc, adjClosed, 0.05 * adjMc, "in-arrears adjustment: closed form vs measure-change MC");
  }

  @Test
  void timingShiftInterpolatesBetweenForwardAndInArrears() {
    double l0 = 0.028;
    double vol = 0.22;
    double tau = 0.5;
    double t = 2.0;

    Adjustment atFixing = TimingAdjustment.liborPaymentShift(l0, vol, tau, t, t);
    Adjustment atNatural = TimingAdjustment.liborPaymentShift(l0, vol, tau, t, t + tau);
    Adjustment inArrears = ConvexityAdjustment.inArrears(l0, vol, tau, t);

    assertEquals(inArrears.adjustment(), atFixing.adjustment(), 1e-12, "pay at fixing == in-arrears");
    assertEquals(0.0, atNatural.adjustment(), 1e-12, "pay at natural date => no adjustment");
    // Pushed one accrual past natural: sign flips.
    Adjustment pushed = TimingAdjustment.liborPaymentShift(l0, vol, tau, t, t + 2 * tau);
    assertEquals(-inArrears.adjustment(), pushed.adjustment(), 1e-12);
  }

  @Test
  void quantoOptionMatchesFxProductsMonteCarlo() {
    QuantoMarket m = new QuantoMarket(100.0, 100.0, 0.22, 0.09, -0.35, 0.03, 0.012);
    double maturity = 1.0;
    double fixedFx = 1.25;

    AnalyticGreeks closed = QuantoAdjustment.quantoOption(OptionType.CALL, m, maturity, fixedFx);
    double mc = Phase1Support.priceAt(m, FxProducts.quantoOption(OptionType.CALL, maturity, 64, fixedFx));

    assertEquals(mc, closed.price(), 0.02 * closed.price() + 1e-3, "quanto option: closed form vs FxProducts MC");

    // Zero correlation collapses to the plain foreign option, converted at fixed FX.
    QuantoMarket zeroCorr = new QuantoMarket(100.0, 100.0, 0.22, 0.09, 0.0, 0.03, 0.012);
    double plain = fixedFx * GeneralizedBsm.of(OptionType.CALL, 100.0, 100.0, maturity,
        zeroCorr.rateDom(), zeroCorr.rateDom() - zeroCorr.rateForeign(), 0.22).price();
    assertEquals(plain, QuantoAdjustment.quantoOption(OptionType.CALL, zeroCorr, maturity, fixedFx).price(),
        1e-9, "zero-correlation quanto == foreign option at fixed FX");
  }

  @Test
  void cmsAdjustmentIsPositiveAndScalesWithVariance() {
    Adjustment low = ConvexityAdjustment.cms(0.03, 0.15, 5.0, 2, 20);
    Adjustment high = ConvexityAdjustment.cms(0.03, 0.30, 5.0, 2, 20);
    assertTrue(low.adjustment() > 0.0, "CMS convexity adjustment is positive");
    assertEquals(4.0, high.adjustment() / low.adjustment(), 0.05, "adjustment scales with vol^2");
  }
}
