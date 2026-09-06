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
package com.nablatensor.quant.estimate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Feature F4: the maximum-likelihood estimators recover the parameters of a
 * simulated process, the closed-form identities hold exactly, and the PCA
 * reconstructs its input matrix.
 */
@Tag("mc")
class EstimateTest {

  /** Simulate a zero-mean GARCH(1,1) return series. */
  private static double[] simulateGarch(double omega, double alpha, double beta, int n, long seed) {
    Random rng = new Random(seed);
    double[] r = new double[n];
    double s2 = omega / (1.0 - alpha - beta);
    for (int t = 0; t < n; t++) {
      double z = rng.nextGaussian();
      r[t] = Math.sqrt(s2) * z;
      s2 = omega + alpha * r[t] * r[t] + beta * s2;
    }
    return r;
  }

  @Test
  void garchFitRecoversSimulatedParameters() {
    double omega = 2.0e-6;
    double alpha = 0.07;
    double beta = 0.92;                 // persistence 0.99, long-run var 2e-4
    double[] returns = simulateGarch(omega, alpha, beta, 6_000, 20260906L);

    Fit fit = Garch11.fit(returns);

    // The MLE fits the sample at least as well as the data-generating parameters.
    double var0 = Ewma.sampleVariance(returns);
    double llTruth = -Garch11.negLogLikelihood(omega, alpha, beta, returns, var0);
    assertTrue(fit.logLik() >= llTruth - 1e-6 * Math.abs(llTruth),
        "fitted log-likelihood not below the truth's");

    assertEquals(alpha + beta, fit.persistence(), 0.02, "persistence");
    assertEquals(alpha, fit.params().alpha(), 0.04, "alpha");
    assertEquals(beta, fit.params().beta(), 0.05, "beta");
    assertEquals(omega / (1.0 - alpha - beta), fit.params().longRunVariance(),
        0.4 * omega / (1.0 - alpha - beta), "long-run variance");

    for (double se : fit.standardErrors()) {
      assertTrue(Double.isFinite(se) && se > 0.0, "standard error finite and positive");
    }
    // The MLE point should be inside a plausible confidence box of the truth.
    assertTrue(Math.abs(fit.params().alpha() - alpha) < 5.0 * fit.standardErrors()[1],
        "alpha within 5 standard errors");
  }

  @Test
  void garchLongRunVarianceMatchesClosedForm() {
    Garch11 g = new Garch11(3.0e-6, 0.06, 0.90);
    assertEquals(3.0e-6 / (1.0 - 0.06 - 0.90), g.longRunVariance(), 1e-18);
    double[] v = g.conditionalVariance(new double[] {0.01, -0.02, 0.005, 0.0, 0.03});
    assertEquals(g.longRunVariance(), v[0], 1e-18, "seeded at the long-run variance");
    assertTrue(v[2] > 0.0);
  }

  @Test
  void ewmaMaximumLikelihoodRecoversDecay() {
    // A near-integrated GARCH is an EWMA with lambda = beta.
    // Near-integrated (alpha + beta = 0.99) so the EWMA-equivalent decay ~ beta.
    double[] returns = simulateGarch(2.0e-6, 0.06, 0.93, 8_000, 424242L);
    double lambda = Ewma.estimateByMaximumLikelihood(returns);

    // The adjoint-gradient optimum must agree with a brute-force grid minimum.
    double gridBest = 0.5;
    double gridNll = Double.POSITIVE_INFINITY;
    for (double l = 0.50; l <= 0.9990; l += 0.0005) {
      double nll = Ewma.negLogLikelihood(returns, l);
      if (nll < gridNll) {
        gridNll = nll;
        gridBest = l;
      }
    }
    assertEquals(gridBest, lambda, 0.01, "MLE decay matches the grid minimum");
    assertTrue(Ewma.negLogLikelihood(returns, lambda) <= gridNll + 1e-6,
        "MLE decay is at least as good as the grid minimum");
  }

  @Test
  void pcaReconstructsAThreeFactorCovariance() {
    // Build cov = sum_k w_k v_k v_k^T with orthonormal-ish loadings.
    double[][] cov = {
        {1.00, 0.60, 0.30},
        {0.60, 1.00, 0.50},
        {0.30, 0.50, 1.00},
    };
    Pca pca = Pca.of(cov);

    // Eigenvalues descending, sum to the trace.
    double[] eig = pca.eigenvalues();
    assertTrue(eig[0] >= eig[1] && eig[1] >= eig[2], "eigenvalues descending");
    assertEquals(3.0, eig[0] + eig[1] + eig[2], 1e-10, "trace preserved");
    assertEquals(eig[0] / 3.0, pca.explainedVariance()[0], 1e-12);

    // V Lambda V^T reconstructs cov.
    double[][] v = pca.loadings();
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        double s = 0.0;
        for (int k = 0; k < 3; k++) {
          s += v[i][k] * eig[k] * v[j][k];
        }
        assertEquals(cov[i][j], s, 1e-9, "reconstruction[" + i + "][" + j + "]");
      }
    }
    // Columns orthonormal.
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        double dot = v[0][a] * v[0][b] + v[1][a] * v[1][b] + v[2][a] * v[2][b];
        assertEquals(a == b ? 1.0 : 0.0, dot, 1e-9, "orthonormality (" + a + "," + b + ")");
      }
    }
  }

  @Test
  void correlationEstimatorBasics() {
    Random rng = new Random(7L);
    int t = 4000;
    double[][] returns = new double[3][t];
    for (int k = 0; k < t; k++) {
      double f = rng.nextGaussian();
      returns[0][k] = f + 0.1 * rng.nextGaussian();
      returns[1][k] = 0.5 * f + 0.5 * rng.nextGaussian();
      returns[2][k] = rng.nextGaussian();          // independent of the factor
    }

    double[][] corr = CorrelationEstimator.sampleCorrelation(returns);
    for (int i = 0; i < 3; i++) {
      assertEquals(1.0, corr[i][i], 1e-12, "unit diagonal");
      for (int j = 0; j < 3; j++) {
        assertEquals(corr[i][j], corr[j][i], 1e-12, "symmetric");
        assertTrue(Math.abs(corr[i][j]) <= 1.0 + 1e-9, "in [-1, 1]");
      }
    }
    assertTrue(corr[0][1] > 0.3, "series 0 and 1 share a factor");
    assertTrue(Math.abs(corr[0][2]) < 0.1, "series 2 uncorrelated with series 0");

    double[][] ew = CorrelationEstimator.ewmaCorrelation(returns, 0.94);
    assertEquals(1.0, ew[1][1], 1e-12);
  }
}
