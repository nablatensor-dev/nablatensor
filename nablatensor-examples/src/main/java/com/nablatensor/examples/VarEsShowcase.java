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
package com.nablatensor.examples;

import com.nablatensor.risk.PnlVector;
import com.nablatensor.risk.ValueAtRisk;
import com.nablatensor.risk.VarBacktest;
import java.util.Locale;
import java.util.Random;

/**
 * Feature F3 — Value at Risk and Expected Shortfall three ways on one book:
 * delta-normal (variance-covariance), delta-gamma with a Cornish-Fisher tail,
 * and historical / full-revaluation from a P&L sample. A coverage backtest of
 * the delta-normal forecast closes the example.
 *
 * <p>The sensitivity vector and the gamma matrix here stand in for what one
 * adjoint reverse sweep of a portfolio tape returns; the point is the tail
 * mathematics on top, not the pricing underneath.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.VarEsShowcase}
 */
public final class VarEsShowcase {

  private VarEsShowcase() {
  }

  // Three risk factors: equity spot return, 5y rate move, EURUSD return.
  private static final double[] DELTA = {2_400_000.0, -18_000_000.0, 1_100_000.0};
  private static final double[][] GAMMA = {
      {-9.0e6, 0.0, 0.0},
      {0.0, -4.0e8, 0.0},
      {0.0, 0.0, -1.5e6},
  };
  // One-day covariance of the three factor moves, built from daily vols
  // (equity 1.1%, 5y rate 1.7bp, EURUSD 0.5%) and a valid correlation matrix
  // (eq/rate -0.15, eq/fx +0.25, rate/fx -0.10).
  private static final double[][] COV = covariance(
      new double[] {1.10e-2, 1.70e-4, 5.00e-3},
      new double[][] {
          {1.00, -0.15, 0.25},
          {-0.15, 1.00, -0.10},
          {0.25, -0.10, 1.00},
      });

  private static double[][] covariance(double[] vol, double[][] corr) {
    int n = vol.length;
    double[][] cov = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        cov[i][j] = corr[i][j] * vol[i] * vol[j];
      }
    }
    return cov;
  }

  public static void main(String[] args) {
    long seed = Long.getLong("seed", 42L);
    int window = Integer.getInteger("window", 4000);

    double dn1 = ValueAtRisk.deltaNormal(DELTA, COV, 0.99, 1.0);
    double dn10 = ValueAtRisk.deltaNormal(DELTA, COV, 0.99, 10.0);
    double dg1 = ValueAtRisk.deltaGammaCornishFisher(DELTA, GAMMA, COV, 0.99, 1.0);
    ValueAtRisk.QuadraticCumulants c = ValueAtRisk.quadraticCumulants(DELTA, GAMMA, COV);

    PnlVector sample = simulatePnl(window, seed);
    double hVar = ValueAtRisk.historical(sample, 0.99);
    double hEs = ValueAtRisk.expectedShortfall(sample, 0.99);

    System.out.printf(Locale.ROOT, "Portfolio VaR / ES  (99%%, reporting currency)%n%n");
    System.out.printf(Locale.ROOT, "  delta-normal        1-day VaR   %,15.0f%n", dn1);
    System.out.printf(Locale.ROOT, "  delta-normal       10-day VaR   %,15.0f%n", dn10);
    System.out.printf(Locale.ROOT, "  delta-gamma (CF)     1-day VaR   %,15.0f   (skew %+.3f, exc. kurt %+.3f)%n",
        dg1, c.skewness(), c.excessKurtosis());
    System.out.printf(Locale.ROOT, "  historical (%d days) 1-day VaR   %,15.0f%n", window, hVar);
    System.out.printf(Locale.ROOT, "  historical (%d days) 1-day ES    %,15.0f%n", window, hEs);

    // Backtest the delta-normal number against a fresh realised series.
    double[] realised = simulatePnl(2000, seed + 1).pnl();
    VarBacktest bt = VarBacktest.of(realised, dn1, 0.99);
    System.out.printf(Locale.ROOT, "%nBacktest of the 1-day delta-normal forecast over %d days:%n", bt.observations());
    System.out.printf(Locale.ROOT, "  exceptions %d  (expected %.1f)%n", bt.exceptions(), bt.expectedExceptions());
    System.out.printf(Locale.ROOT, "  Kupiec POF p-value            %.3f%n", bt.kupiecPValue());
    System.out.printf(Locale.ROOT, "  Christoffersen independence   %.3f%n", bt.christoffersenPValue());
    System.out.printf(Locale.ROOT, "  conditional coverage p-value  %.3f  =>  %s%n",
        bt.conditionalCoveragePValue(), bt.rejectedAt(0.05) ? "REJECT" : "accept");
  }

  /** Draw a P&L sample from the quadratic model delta' x + 0.5 x' Gamma x, x ~ N(0, COV). */
  private static PnlVector simulatePnl(int n, long seed) {
    double[][] l = cholesky(COV);
    Random rng = new Random(seed);
    double[] pnl = new double[n];
    for (int t = 0; t < n; t++) {
      double[] z = {rng.nextGaussian(), rng.nextGaussian(), rng.nextGaussian()};
      double[] x = new double[3];
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j <= i; j++) {
          x[i] += l[i][j] * z[j];
        }
      }
      double lin = DELTA[0] * x[0] + DELTA[1] * x[1] + DELTA[2] * x[2];
      double quad = 0.0;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          quad += 0.5 * GAMMA[i][j] * x[i] * x[j];
        }
      }
      pnl[t] = lin + quad;
    }
    return new PnlVector(pnl);
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
