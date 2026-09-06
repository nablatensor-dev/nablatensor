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

import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Feature F5: the multi-curve bootstrap reprices its calibrating instruments,
 * collapses to a single curve when the forecast quotes are the OIS par rates,
 * and its adjoint Jacobian matches a central bump and is block lower-triangular.
 */
class MultiCurveBootstrapTest {

  private static final double[] OIS = {0.0300, 0.0315, 0.0325, 0.0332, 0.0338};   // 1y..5y
  private static final String T3M = "3M";

  private static MultiCurveBootstrap.Result bootstrap(double[] ois, double[] fc) {
    MultiCurveBootstrap.Builder b = MultiCurveBootstrap.builder();
    for (int i = 0; i < ois.length; i++) {
      b.oisSwap(i + 1, ois[i]);
    }
    for (int i = 0; i < fc.length; i++) {
      b.forecastSwap(T3M, i + 1, fc[i]);
    }
    return b.build().solve();
  }

  @Test
  void repricesEveryCalibratingInstrument() {
    double[] fc = {0.0325, 0.0340, 0.0350, 0.0357, 0.0363};   // 25bp tenor basis, roughly
    CurveSet cs = bootstrap(OIS, fc).curves();

    for (int n = 1; n <= 5; n++) {
      double oisPar = (1.0 - cs.df(n)) / cs.annuity(n);
      assertEquals(OIS[n - 1], oisPar, 1e-9, "OIS swap " + n + "y reprices");
      assertEquals(fc[n - 1], cs.parSwapRate(T3M, n), 1e-9, "3M swap " + n + "y reprices");
    }
  }

  @Test
  void collapsesToOneCurveWhenForecastQuotesAreOisParRates() {
    // First bootstrap OIS alone, read its own multi-curve-consistent par rates,
    // feed those as the forecast quotes: the forecast curve must equal the OIS curve.
    CurveSet oisOnly = bootstrap(OIS, OIS).curves();
    double[] fcQuotes = new double[5];
    for (int n = 1; n <= 5; n++) {
      // par rate of a swap that forecasts and discounts on the SAME OIS curve
      double floatPv = 0.0;
      double annuity = 0.0;
      double prev = 1.0;
      for (int i = 1; i <= n; i++) {
        double p = oisOnly.df(i);
        floatPv += (prev / p - 1.0) * p;
        annuity += p;
        prev = p;
      }
      fcQuotes[n - 1] = floatPv / annuity;
    }

    CurveSet cs = bootstrap(OIS, fcQuotes).curves();
    for (int n = 1; n <= 5; n++) {
      assertEquals(cs.discount().zeroRate(n), cs.forecast(T3M).zeroRate(n), 1e-9,
          "forecast zero == OIS zero at " + n + "y");
    }
  }

  @Test
  void adjointJacobianMatchesCentralBump() {
    double[] fc = {0.0326, 0.0341, 0.0351, 0.0358, 0.0364};
    MultiCurveBootstrap builder = builder(OIS, fc);
    try (MultiCurveBootstrap.Kernel kernel = builder.kernel()) {
      MultiCurveBootstrap.Result base = kernel.evaluate(Map.of());
      double[][] jac = base.jacobian();
      var labels = base.quoteLabels();
      var zeroLabels = base.zeroLabels();
      double h = 1e-6;

      for (int col = 0; col < labels.size(); col++) {
        double[] baseZ = zeros(base);
        // one compiled kernel, re-evaluated at the shifted quote — no re-record.
        double[] up = zeros(kernel.evaluate(Map.of(labels.get(col), quote(OIS, fc, col) + h)));
        double[] dn = zeros(kernel.evaluate(Map.of(labels.get(col), quote(OIS, fc, col) - h)));
        for (int row = 0; row < zeroLabels.size(); row++) {
          double bump = (up[row] - dn[row]) / (2 * h);
          assertEquals(bump, jac[row][col], 1e-5 * (1 + Math.abs(bump)),
              "J[" + zeroLabels.get(row) + "][" + labels.get(col) + "]");
          assertTrue(Double.isFinite(baseZ[row]));
        }
      }
    }
  }

  private static double quote(double[] ois, double[] fc, int col) {
    return col < ois.length ? ois[col] : fc[col - ois.length];
  }

  private static MultiCurveBootstrap builder(double[] ois, double[] fc) {
    MultiCurveBootstrap.Builder b = MultiCurveBootstrap.builder();
    for (int i = 0; i < ois.length; i++) {
      b.oisSwap(i + 1, ois[i]);
    }
    for (int i = 0; i < fc.length; i++) {
      b.forecastSwap(T3M, i + 1, fc[i]);
    }
    return b.build();
  }

  private static double[] zeros(MultiCurveBootstrap.Result r) {
    CurveSet cs = r.curves();
    double[] z = new double[r.zeroLabels().size()];
    int k = 0;
    for (double p : cs.discount().pillars()) {
      z[k++] = cs.discount().zeroRate(p);
    }
    for (double p : cs.forecast(T3M).pillars()) {
      z[k++] = cs.forecast(T3M).zeroRate(p);
    }
    return z;
  }

  @Test
  void jacobianIsBlockLowerTriangular() {
    double[] fc = {0.0327, 0.0342, 0.0352, 0.0359, 0.0365};
    MultiCurveBootstrap.Result r = bootstrap(OIS, fc);
    int nOis = OIS.length;

    for (int row = 0; row < nOis; row++) {
      for (int col = nOis; col < r.quoteLabels().size(); col++) {
        assertEquals(0.0, r.jacobian()[row][col], 1e-12,
            "OIS zero must not depend on a forecast quote");
      }
    }
    // A forecast zero depends on the OIS quotes that move its discounting.
    double crossTerm = 0.0;
    for (int col = 0; col < nOis; col++) {
      crossTerm += Math.abs(r.jacobian()[nOis + 4][col]);   // 5y forecast zero vs OIS quotes
    }
    assertTrue(crossTerm > 1e-6, "5y forecast zero should react to OIS quotes, got " + crossTerm);
  }

}
