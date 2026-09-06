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
package com.nablatensor.risk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.quant.analytic.Normal;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Feature F3: the VaR / ES estimators agree with their analytic values on a
 * Gaussian sample, delta-normal is exact on a linear book, delta-gamma matches
 * a Monte-Carlo of the quadratic P&L, and the backtest accepts a calibrated
 * forecast while rejecting a mis-calibrated one.
 */
class ValueAtRiskTest {

  private static final double Z99 = Normal.inverseCdf(0.99);   // ~2.326

  @Test
  void historicalVarAndEsMatchTheGaussianClosedForm() {
    double sigma = 0.02;
    Random rng = new Random(11L);
    double[] pnl = new double[400_000];
    for (int i = 0; i < pnl.length; i++) {
      pnl[i] = sigma * rng.nextGaussian();
    }
    PnlVector sample = new PnlVector(pnl);

    double var = ValueAtRisk.historical(sample, 0.99);
    double es = ValueAtRisk.expectedShortfall(sample, 0.99);

    assertEquals(Z99 * sigma, var, 0.03 * Z99 * sigma, "historical VaR ~ z * sigma");
    // ES of a normal at 99%: sigma * phi(z) / (1 - alpha)
    double esClosed = sigma * Normal.pdf(Z99) / 0.01;
    assertEquals(esClosed, es, 0.04 * esClosed, "historical ES ~ sigma phi(z) / (1-alpha)");
    assertTrue(es > var, "ES exceeds VaR");
  }

  @Test
  void deltaNormalIsExactOnALinearBook() {
    // Two factors, known covariance; VaR = z * sqrt(s' Sigma s).
    double[] s = {150.0, -80.0};
    double[][] cov = {
        {4.0e-4, 1.2e-4},
        {1.2e-4, 9.0e-4},
    };
    double quad = 0.0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        quad += s[i] * cov[i][j] * s[j];
      }
    }
    double expected = Z99 * Math.sqrt(quad);
    assertEquals(expected, ValueAtRisk.deltaNormal(s, cov, 0.99, 1.0), 1e-9);
    // 10-day figure scales by sqrt(10).
    assertEquals(expected * Math.sqrt(10.0), ValueAtRisk.deltaNormal(s, cov, 0.99, 10.0), 1e-9);
  }

  @Test
  void deltaNormalMatchesHistoricalForALinearPortfolio() {
    double[] s = {100.0, 50.0};
    double[][] cov = {
        {2.5e-4, -0.5e-4},
        {-0.5e-4, 1.0e-4},
    };
    double[][] chol = cholesky(cov);
    Random rng = new Random(99L);
    double[] pnl = new double[300_000];
    for (int k = 0; k < pnl.length; k++) {
      double z0 = rng.nextGaussian();
      double z1 = rng.nextGaussian();
      double x0 = chol[0][0] * z0;
      double x1 = chol[1][0] * z0 + chol[1][1] * z1;
      pnl[k] = s[0] * x0 + s[1] * x1;
    }
    double hist = ValueAtRisk.historical(new PnlVector(pnl), 0.99);
    double dn = ValueAtRisk.deltaNormal(s, cov, 0.99, 1.0);
    assertEquals(dn, hist, 0.03 * dn, "linear portfolio: historical ~ delta-normal");
  }

  @Test
  void deltaGammaReducesToDeltaNormalWhenGammaIsZero() {
    double[] delta = {120.0, -60.0};
    double[][] sigma = {
        {3.0e-4, 0.8e-4},
        {0.8e-4, 5.0e-4},
    };
    double[][] zeroGamma = new double[2][2];
    double dn = ValueAtRisk.deltaNormal(delta, sigma, 0.99, 1.0);
    double dg = ValueAtRisk.deltaGammaCornishFisher(delta, zeroGamma, sigma, 0.99, 1.0);
    assertEquals(dn, dg, 1e-7, "delta-gamma with Gamma=0 is delta-normal");
  }

  @Test
  void deltaGammaCumulantsMatchMonteCarlo() {
    double[] delta = {80.0, 40.0};
    double[][] gamma = {
        {-1500.0, 200.0},
        {200.0, -900.0},
    };
    double[][] sigma = {
        {4.0e-4, 1.0e-4},
        {1.0e-4, 2.5e-4},
    };
    ValueAtRisk.QuadraticCumulants c = ValueAtRisk.quadraticCumulants(delta, gamma, sigma);

    double[][] chol = cholesky(sigma);
    Random rng = new Random(2024L);
    int n = 2_000_000;
    double[] q = new double[n];
    double m1 = 0.0;
    for (int t = 0; t < n; t++) {
      double z0 = rng.nextGaussian();
      double z1 = rng.nextGaussian();
      double x0 = chol[0][0] * z0;
      double x1 = chol[1][0] * z0 + chol[1][1] * z1;
      double lin = delta[0] * x0 + delta[1] * x1;
      double quad = 0.5 * (gamma[0][0] * x0 * x0 + 2 * gamma[0][1] * x0 * x1 + gamma[1][1] * x1 * x1);
      q[t] = lin + quad;
      m1 += q[t];
    }
    m1 /= n;
    double m2 = 0.0;
    double m3 = 0.0;
    double m4 = 0.0;
    for (double v : q) {
      double d = v - m1;
      m2 += d * d;
      m3 += d * d * d;
      m4 += d * d * d * d;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;
    double sampleSkew = m3 / Math.pow(m2, 1.5);
    double sampleKurt = m4 / (m2 * m2) - 3.0;

    assertEquals(m1, c.mean(), 0.02 * Math.abs(m1) + 1e-9, "mean");
    assertEquals(m2, c.variance(), 0.02 * m2, "variance");
    assertEquals(sampleSkew, c.skewness(), 0.05 + 0.1 * Math.abs(sampleSkew), "skewness");
    assertEquals(sampleKurt, c.excessKurtosis(), 0.15 + 0.15 * Math.abs(sampleKurt), "excess kurtosis");

    // The Cornish-Fisher VaR should be near the empirical quantile of the loss.
    java.util.Arrays.sort(q);
    double empiricalVar = -q[(int) (0.01 * n)];
    double cfVar = ValueAtRisk.deltaGammaCornishFisher(delta, gamma, sigma, 0.99, 1.0);
    assertEquals(empiricalVar, cfVar, 0.12 * empiricalVar, "CF VaR near the empirical quantile");
  }

  @Test
  void backtestAcceptsCalibratedRejectsMiscalibrated() {
    double sigma = 0.015;
    Random rng = new Random(7L);
    int n = 3000;
    double[] pnl = new double[n];
    for (int t = 0; t < n; t++) {
      pnl[t] = sigma * rng.nextGaussian();
    }

    VarBacktest good = VarBacktest.of(pnl, Z99 * sigma, 0.99);
    assertEquals(n * 0.01, good.expectedExceptions(), 1e-9);
    assertTrue(good.exceptions() >= 12 && good.exceptions() <= 48,
        "calibrated forecast: exceptions near 30, got " + good.exceptions());
    assertFalse(good.rejectedAt(0.01), "calibrated forecast not rejected");

    VarBacktest bad = VarBacktest.of(pnl, 0.8 * sigma, 0.99);   // far too small
    assertTrue(bad.exceptions() > 100, "under-forecast: many exceptions");
    assertTrue(bad.rejectedAt(0.01), "under-forecast rejected by conditional coverage");
  }

  @Test
  void backtestExceptionCountIsExact() {
    double[] pnl = {0.01, -0.05, 0.02, -0.20, -0.011, 0.0};
    double[] forecast = {0.03, 0.03, 0.03, 0.03, 0.01, 0.03};
    // losses:            -0.01 0.05  -0.02 0.20   0.011  0.0
    // exception if loss > forecast: day1 no, day2 yes, day3 no, day4 yes, day5 yes, day6 no
    VarBacktest bt = VarBacktest.of(pnl, forecast, 0.99);
    assertEquals(3, bt.exceptions());
    assertEquals(6, bt.observations());
  }

  private static double[][] cholesky(double[][] a) {
    int n = a.length;
    double[][] l = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++) {
        double s = a[i][j];
        for (int k = 0; k < j; k++) {
          s -= l[i][k] * l[j][k];
        }
        l[i][j] = i == j ? Math.sqrt(s) : s / l[j][j];
      }
    }
    return l;
  }
}
