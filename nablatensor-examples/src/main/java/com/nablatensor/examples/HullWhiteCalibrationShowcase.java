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

import com.nablatensor.quant.HullWhiteAnalytic;
import com.nablatensor.quant.HullWhiteCalibration;
import com.nablatensor.quant.YieldCurve;
import java.util.List;
import java.util.Locale;

/**
 * Feature F6 — the term-structure Hull-White model. Given today's curve, the
 * bond reconstitution fits it exactly (no separate {@code theta(t)} solve), the
 * caplet and Jamshidian swaption price in closed form, and {@code (a, sigma)}
 * are calibrated to a co-terminal swaption grid.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.HullWhiteCalibrationShowcase}
 */
public final class HullWhiteCalibrationShowcase {

  private HullWhiteCalibrationShowcase() {
  }

  public static void main(String[] args) {
    // Upward-sloping curve, zeros 2.6% -> 3.5% over 1y..12y.
    int n = 12;
    double[] pillars = new double[n];
    double[] zeros = new double[n];
    for (int i = 0; i < n; i++) {
      pillars[i] = i + 1.0;
      zeros[i] = 0.026 + 0.009 * (i / (n - 1.0));
    }
    YieldCurve curve = new YieldCurve(pillars, zeros);

    // A co-terminal ~10y diagonal. The ATM normal vols come from a reference
    // Hull-White model (a=0.10, sigma=95bp) with a small idiosyncratic bump per
    // node, so the market is close to — but not exactly — one-factor consistent.
    double[] expiries = {1, 2, 3, 4, 5, 7};
    int[] tenors = {9, 8, 7, 6, 5, 3};
    double accrual = 1.0;
    HullWhiteAnalytic reference = HullWhiteAnalytic.of(curve, 0.10, 0.0095);
    double[] bump = {1.03, 0.99, 1.01, 1.00, 0.98, 1.02};
    double[] normalVols = new double[expiries.length];
    for (int i = 0; i < expiries.length; i++) {
      int periods = tenors[i];
      double ann = 0.0;
      for (int j = 1; j <= periods; j++) {
        ann += accrual * curve.discountFactor(expiries[i] + j * accrual);
      }
      double fwd = (curve.discountFactor(expiries[i])
          - curve.discountFactor(expiries[i] + periods * accrual)) / ann;
      double px = reference.payerSwaption(expiries[i], accrual, periods, fwd);
      normalVols[i] = bump[i] * px / (ann * Math.sqrt(expiries[i] / (2.0 * Math.PI)));
    }

    List<HullWhiteCalibration.SwaptionQuote> quotes =
        HullWhiteCalibration.grid(expiries, tenors, accrual, normalVols);

    long t0 = System.nanoTime();
    HullWhiteCalibration.Result r = HullWhiteCalibration.calibrate(curve, quotes, 0.03, 0.02);
    double ms = (System.nanoTime() - t0) / 1e6;

    System.out.printf(Locale.ROOT, "Hull-White (a, sigma) calibration to %d co-terminal swaptions  (%d iters, %.0f ms)%n%n",
        quotes.size(), r.iterations(), ms);
    System.out.printf(Locale.ROOT, "  a      = %.4f%n", r.a());
    System.out.printf(Locale.ROOT, "  sigma  = %.5f  (%.1f bp/yr)%n", r.sigma(), r.sigma() * 1e4);
    System.out.printf(Locale.ROOT, "  price RMSE = %.3e   converged=%b%n%n", r.rmsePrice(), r.converged());

    System.out.printf(Locale.ROOT, "  %-6s %-6s %14s %14s%n", "expiry", "tenor", "target px", "model px");
    for (int k = 0; k < quotes.size(); k++) {
      System.out.printf(Locale.ROOT, "  %-6.0f %-6d %14.6f %14.6f%n",
          expiries[k], tenors[k], r.targetPrices()[k], r.modelPrices()[k]);
    }

    // Use the fitted model: a 2y-into-5y caplet strip and the curve reprice.
    HullWhiteAnalytic hw = r.model(curve);
    double cap = hw.cap(2.0, 1.0, 5, 0.033);
    System.out.printf(Locale.ROOT, "%n5-period annual cap @ 3.30%%, first reset 2y:  %.6f%n", cap);
    System.out.printf(Locale.ROOT, "curve reprice check  P(0,10) model %.8f  curve %.8f%n",
        hw.bondReconstitution(0.0, 10.0, hw.r0()), curve.discountFactor(10.0));
  }
}
