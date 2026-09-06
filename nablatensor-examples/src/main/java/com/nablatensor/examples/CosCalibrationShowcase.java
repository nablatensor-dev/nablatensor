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

import com.nablatensor.quant.OptionType;
import com.nablatensor.quant.analytic.GeneralizedBsm;
import com.nablatensor.quant.transform.BsmCf;
import com.nablatensor.quant.transform.CosMethod;
import com.nablatensor.quant.transform.HestonCf;
import com.nablatensor.quant.transform.HestonCosCalibrator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Feature F13 — the Fang-Oosterlee COS method. Its accuracy against Black-Scholes
 * is shown first, then a Heston implied-vol surface is generated, perturbed, and
 * calibrated back — all deterministic, the whole surface fit in a fraction of a
 * second.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.CosCalibrationShowcase}
 */
public final class CosCalibrationShowcase {

  private CosCalibrationShowcase() {
  }

  public static void main(String[] args) {
    double s = 100;
    double r = 0.02;

    // COS vs Black-Scholes: spectral accuracy.
    BsmCf bsm = new BsmCf(r, 0.2);
    double maxErr = 0.0;
    for (double k : new double[] {70, 85, 100, 115, 130}) {
      double cos = CosMethod.price(bsm, OptionType.CALL, s, k, r, 1.0);
      double closed = GeneralizedBsm.of(OptionType.CALL, s, k, 1.0, r, 0.0, 0.2).price();
      maxErr = Math.max(maxErr, Math.abs(cos - closed));
    }
    System.out.printf(Locale.ROOT, "COS vs Black-Scholes: max abs error over 5 strikes = %.2e%n%n", maxErr);

    // Generate a Heston surface, then calibrate to it (with a slight per-node bump).
    HestonCf truth = new HestonCf(r, 0.045, 2.0, 0.05, 0.55, -0.65);
    double[] maturities = {0.5, 1.0, 2.0};
    double[] strikes = {85, 95, 105, 115};
    List<HestonCosCalibrator.Quote> quotes = new ArrayList<>();
    double bump = 1.0;
    for (double t : maturities) {
      for (double k : strikes) {
        double px = CosMethod.price(truth, OptionType.CALL, s, k, r, t);
        quotes.add(new HestonCosCalibrator.Quote(k, t, px * bump));
        bump = bump == 1.0 ? 1.004 : 1.0;
      }
    }

    long t0 = System.nanoTime();
    HestonCosCalibrator.Result fit = HestonCosCalibrator.calibrate(
        s, r, quotes, new double[] {0.04, 1.0, 0.04, 0.3, -0.3});
    double ms = (System.nanoTime() - t0) / 1e6;

    System.out.printf(Locale.ROOT, "Heston surface calibration (%d quotes, %d iters, %.0f ms)%n", quotes.size(),
        fit.iterations(), ms);
    System.out.printf(Locale.ROOT, "  %-8s %10s %10s%n", "param", "true", "fitted");
    System.out.printf(Locale.ROOT, "  %-8s %10.4f %10.4f%n", "v0", 0.045, fit.v0());
    System.out.printf(Locale.ROOT, "  %-8s %10.4f %10.4f%n", "kappa", 2.0, fit.kappa());
    System.out.printf(Locale.ROOT, "  %-8s %10.4f %10.4f%n", "theta", 0.05, fit.theta());
    System.out.printf(Locale.ROOT, "  %-8s %10.4f %10.4f%n", "xi", 0.55, fit.xi());
    System.out.printf(Locale.ROOT, "  %-8s %10.4f %10.4f%n", "rho", -0.65, fit.rho());
    System.out.printf(Locale.ROOT, "  price RMSE = %.3e   converged=%b%n", fit.rmse(), fit.converged());
  }
}
