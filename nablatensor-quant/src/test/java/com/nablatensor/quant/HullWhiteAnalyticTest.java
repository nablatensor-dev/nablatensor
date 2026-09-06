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

import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Feature F6: the term-structure Hull-White model reprices the initial curve
 * exactly, satisfies bond-option and swaption parity, agrees with a direct
 * Monte-Carlo of the short rate, has a well-behaved Ho-Lee limit, and its
 * {@code (a, sigma)} calibration recovers the parameters a grid was generated
 * from.
 */
class HullWhiteAnalyticTest {

  /** A mildly upward-sloping curve: zero rates 2.5% -> 3.4% over 1y..12y. */
  private static YieldCurve curve() {
    int n = 12;
    double[] pillars = new double[n];
    double[] zeros = new double[n];
    for (int i = 0; i < n; i++) {
      pillars[i] = i + 1.0;
      zeros[i] = 0.025 + 0.009 * (i / (n - 1.0));
    }
    return new YieldCurve(pillars, zeros);
  }

  @Test
  void repricesTheInitialCurveExactly() {
    YieldCurve c = curve();
    HullWhiteAnalytic hw = HullWhiteAnalytic.of(c, 0.08, 0.011);
    for (double t = 0.5; t <= 12.0; t += 0.5) {
      assertEquals(c.discountFactor(t), hw.bondReconstitution(0.0, t, hw.r0()), 1e-9,
          "P(0," + t + ") reconstituted from r0");
    }
  }

  @Test
  void zeroBondOptionParity() {
    HullWhiteAnalytic hw = HullWhiteAnalytic.of(curve(), 0.05, 0.012);
    double t = 2.0;
    double s = 7.0;
    double k = 0.80;
    double call = hw.zeroBondCall(t, s, k);
    double put = hw.zeroBondPut(t, s, k);
    // call - put = P(0,S) - K P(0,T)
    double fwd = curve().discountFactor(s) - k * curve().discountFactor(t);
    assertEquals(fwd, call - put, 1e-12, "ZCB option put-call parity");
    assertTrue(call > 0 && put > 0, "both legs positive");
  }

  @Test
  void swaptionPayerReceiverParityAndAtmEquality() {
    YieldCurve c = curve();
    HullWhiteAnalytic hw = HullWhiteAnalytic.of(c, 0.06, 0.010);
    double expiry = 3.0;
    double accrual = 1.0;
    int periods = 5;

    double annuity = 0.0;
    for (int i = 1; i <= periods; i++) {
      annuity += accrual * c.discountFactor(expiry + i * accrual);
    }
    double fwdSwap = (c.discountFactor(expiry) - c.discountFactor(expiry + periods * accrual)) / annuity;

    double k = 0.033;
    double payer = hw.payerSwaption(expiry, accrual, periods, k);
    double receiver = hw.receiverSwaption(expiry, accrual, periods, k);
    // payer - receiver = annuity * (fwdSwap - K)
    assertEquals(annuity * (fwdSwap - k), payer - receiver, 1e-9, "payer-receiver parity");

    double payerAtm = hw.payerSwaption(expiry, accrual, periods, fwdSwap);
    double receiverAtm = hw.receiverSwaption(expiry, accrual, periods, fwdSwap);
    assertEquals(payerAtm, receiverAtm, 1e-9, "ATM payer == receiver");
    assertTrue(payerAtm > 0, "positive time value");
  }

  @Test
  void analyticSwaptionAgreesWithShortRateMonteCarlo() {
    YieldCurve c = curve();
    double a = 0.07;
    double sigma = 0.011;
    HullWhiteAnalytic hw = HullWhiteAnalytic.of(c, a, sigma);
    double expiry = 2.0;
    double accrual = 1.0;
    int periods = 4;
    double strike = 0.033;

    double analytic = hw.payerSwaption(expiry, accrual, periods, strike);

    // Risk-neutral MC of the OU factor x: dx = -a x dt + sigma dW, r(t) = f^M(0,t) + x + phi(t).
    int steps = 100;
    int paths = 400_000;
    double dt = expiry / steps;
    double sqrtDt = Math.sqrt(dt);
    Random rng = new Random(20260906L);
    double sum = 0.0;
    for (int p = 0; p < paths; p++) {
      double x = 0.0;
      double intR = 0.0;
      for (int step = 0; step < steps; step++) {
        double tMid = (step + 0.5) * dt;
        double phi = sigma * sigma / (2 * a * a) * Math.pow(1 - Math.exp(-a * tMid), 2);
        double rMid = hw.instantaneousForward(tMid) + x + phi;
        intR += rMid * dt;
        x += -a * x * dt + sigma * sqrtDt * rng.nextGaussian();
      }
      double phiT = sigma * sigma / (2 * a * a) * Math.pow(1 - Math.exp(-a * expiry), 2);
      double rT = hw.instantaneousForward(expiry) + x + phiT;
      double annuity = 0.0;
      double lastBond = 1.0;
      for (int i = 1; i <= periods; i++) {
        double pb = hw.bondReconstitution(expiry, expiry + i * accrual, rT);
        annuity += accrual * pb;
        lastBond = pb;
      }
      double swapRate = (1.0 - lastBond) / annuity;
      double payoff = annuity * Math.max(swapRate - strike, 0.0);
      sum += payoff * Math.exp(-intR);
    }
    double mc = sum / paths;
    assertEquals(analytic, mc, 0.02 * analytic + 1e-5, "analytic swaption vs short-rate MC");
  }

  @Test
  void hoLeeLimitIsFinite() {
    HullWhiteAnalytic ho = HullWhiteAnalytic.of(curve(), 1e-9, 0.010);
    assertEquals(5.0, ho.bFactor(0.0, 5.0), 1e-4, "B(0,5) -> 5 as a -> 0");
    double swaption = ho.payerSwaption(2.0, 1.0, 4, 0.033);
    assertTrue(Double.isFinite(swaption) && swaption > 0, "Ho-Lee swaption finite and positive");
    assertEquals(curve().discountFactor(6.0), ho.bondReconstitution(0.0, 6.0, ho.r0()), 1e-9);
  }

  @Test
  void calibrationRecoversGeneratingParameters() {
    YieldCurve c = curve();
    double aStar = 0.09;
    double sigmaStar = 0.0095;
    HullWhiteAnalytic truth = HullWhiteAnalytic.of(c, aStar, sigmaStar);

    double[] expiries = {1, 2, 3, 5, 7};
    int[] tenors = {5, 4, 3, 5, 3};
    double accrual = 1.0;

    // Build quotes whose Bachelier-implied normal vol reproduces the truth model's price.
    double[] normalVols = new double[expiries.length];
    for (int k = 0; k < expiries.length; k++) {
      int periods = tenors[k];
      double annuity = 0.0;
      for (int i = 1; i <= periods; i++) {
        annuity += accrual * c.discountFactor(expiries[k] + i * accrual);
      }
      double fwd = (c.discountFactor(expiries[k]) - c.discountFactor(expiries[k] + periods * accrual)) / annuity;
      double price = truth.payerSwaption(expiries[k], accrual, periods, fwd);
      // ATM Bachelier: price = annuity * sigmaN * sqrt(T / (2 pi))
      normalVols[k] = price / (annuity * Math.sqrt(expiries[k] / (2.0 * Math.PI)));
    }

    List<HullWhiteCalibration.SwaptionQuote> quotes =
        HullWhiteCalibration.grid(expiries, tenors, accrual, normalVols);
    HullWhiteCalibration.Result r = HullWhiteCalibration.calibrate(c, quotes, 0.03, 0.02);

    assertTrue(r.converged(), "calibration converged");
    assertEquals(aStar, r.a(), 0.01, "mean reversion recovered");
    assertEquals(sigmaStar, r.sigma(), 5e-4, "volatility recovered");
    assertTrue(r.rmsePrice() < 1e-6, "price RMSE tiny, got " + r.rmsePrice());
  }
}
