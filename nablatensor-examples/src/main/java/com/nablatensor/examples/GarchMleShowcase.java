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

import com.nablatensor.quant.estimate.Ewma;
import com.nablatensor.quant.estimate.Fit;
import com.nablatensor.quant.estimate.Garch11;
import com.nablatensor.quant.estimate.Pca;
import java.util.Locale;
import java.util.Random;

/**
 * Feature F4 — estimating volatilities and correlations. A GARCH(1,1) return
 * series is simulated, then its parameters are recovered by Gaussian maximum
 * likelihood: the negative log-likelihood is <em>recorded</em> once and every
 * optimiser step reads the exact score vector from one adjoint sweep. The
 * fitted persistence and long-run volatility are printed next to the
 * data-generating values, with asymptotic standard errors. A short PCA of a
 * synthetic three-factor curve closes the example.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.GarchMleShowcase}
 */
public final class GarchMleShowcase {

  private GarchMleShowcase() {
  }

  public static void main(String[] args) {
    int n = Integer.getInteger("obs", 8_000);
    long seed = Long.getLong("seed", 20260906L);

    double omega = 2.0e-6;
    double alpha = 0.07;
    double beta = 0.92;                 // persistence 0.99
    double[] returns = simulate(omega, alpha, beta, n, seed);

    long t0 = System.nanoTime();
    Fit fit = Garch11.fit(returns);
    double ms = (System.nanoTime() - t0) / 1e6;

    double[] se = fit.standardErrors();
    System.out.printf(Locale.ROOT, "GARCH(1,1) MLE on %,d observations  (%d iterations, %.0f ms)%n%n", n,
        fit.iterations(), ms);
    System.out.printf(Locale.ROOT, "%-14s %12s %12s %12s%n", "", "true", "fitted", "std error");
    row("omega", omega, fit.params().omega(), se[0]);
    row("alpha", alpha, fit.params().alpha(), se[1]);
    row("beta", beta, fit.params().beta(), se[2]);
    System.out.printf(Locale.ROOT, "%-14s %12.4f %12.4f%n", "persistence", alpha + beta, fit.persistence());
    System.out.printf(Locale.ROOT, "%-14s %12.4f %12.4f%n", "ann. long-run vol",
        annualVol(omega / (1 - alpha - beta)), annualVol(fit.params().longRunVariance()));

    double lambda = Ewma.estimateByMaximumLikelihood(returns);
    System.out.printf(Locale.ROOT, "%nRiskMetrics EWMA decay (MLE): lambda = %.4f%n", lambda);

    // A synthetic level/slope/curvature covariance for a 5-tenor curve.
    double[][] cov = curveCovariance();
    Pca pca = Pca.of(cov);
    System.out.printf(Locale.ROOT, "%nPCA of a 5-tenor curve covariance — variance explained:%n");
    double cum = 0.0;
    for (int i = 0; i < pca.explainedVariance().length; i++) {
      cum += pca.explainedVariance()[i];
      System.out.printf(Locale.ROOT, "  PC%d  %6.2f%%   (cumulative %6.2f%%)%n",
          i + 1, 100 * pca.explainedVariance()[i], 100 * cum);
    }
  }

  private static double[] simulate(double w, double a, double b, int n, long seed) {
    Random rng = new Random(seed);
    double[] r = new double[n];
    double s2 = w / (1.0 - a - b);
    for (int t = 0; t < n; t++) {
      r[t] = Math.sqrt(s2) * rng.nextGaussian();
      s2 = w + a * r[t] * r[t] + b * s2;
    }
    return r;
  }

  private static double[][] curveCovariance() {
    // level (flat), slope (linear), curvature (bowl) factors + a little idiosyncratic.
    double[] level = {1, 1, 1, 1, 1};
    double[] slope = {-2, -1, 0, 1, 2};
    double[] curve = {2, -1, -2, -1, 2};
    double[][] cov = new double[5][5];
    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 5; j++) {
        cov[i][j] = 6.0e-5 * level[i] * level[j]
            + 1.2e-5 * slope[i] * slope[j]
            + 3.0e-6 * curve[i] * curve[j]
            + (i == j ? 2.0e-6 : 0.0);
      }
    }
    return cov;
  }

  private static double annualVol(double dailyVariance) {
    return Math.sqrt(dailyVariance * 252.0);
  }

  private static void row(String name, double truth, double fitted, double se) {
    System.out.printf(Locale.ROOT, "%-14s %12.3e %12.3e %12.2e%n", name, truth, fitted, se);
  }
}
