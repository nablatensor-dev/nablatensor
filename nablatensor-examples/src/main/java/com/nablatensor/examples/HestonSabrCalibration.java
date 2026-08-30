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

import com.nablatensor.engine.SDouble;
import com.nablatensor.quant.Calibrator;
import com.nablatensor.quant.SabrHagan;
import java.util.Locale;

/**
 * The Phase-1 killer demo: fit SABR {@code (alpha, rho, nu)} to a volatility
 * smile with an adjoint gradient. One recorded objective, one reverse sweep per
 * iteration, a handful of iterations.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.HestonSabrCalibration}
 */
public final class HestonSabrCalibration {

  private static final double F = 0.05;
  private static final double T = 1.0;
  private static final double BETA = 0.5;
  private static final double[] STRIKES = {0.030, 0.038, 0.045, 0.055, 0.062, 0.070};

  private HestonSabrCalibration() {
  }

  public static void main(String[] args) {
    double trueAlpha = 0.284;
    double trueRho = -0.31;
    double trueNu = 0.57;

    double[] target = new double[STRIKES.length];
    for (int i = 0; i < STRIKES.length; i++) {
      target[i] = SabrHagan.blackVol(trueAlpha, BETA, trueRho, trueNu, F, STRIKES[i], T);
    }

    long t0 = System.nanoTime();
    Calibrator.Result r = Calibrator.of(rec -> {
          SDouble alpha = rec.input("alpha", 0.20);
          SDouble rho = rec.input("rho", 0.0);
          SDouble nu = rec.input("nu", 0.30);
          SDouble beta = rec.constant(BETA);
          SDouble sse = rec.constant(0.0);
          for (int i = 0; i < STRIKES.length; i++) {
            SDouble d = SabrHagan.blackVol(rec, alpha, beta, rho, nu, F, STRIKES[i], T).sub(target[i]);
            sse = sse.add(d.mul(d));
          }
          rec.output(sse);
        })
        .parameter("alpha", 0.20, 1e-4, 2.0)
        .parameter("rho", 0.0, -0.999, 0.999)
        .parameter("nu", 0.30, 1e-4, 5.0)
        .maxIterations(80)
        .tolerance(1e-12)
        .solve();
    double ms = (System.nanoTime() - t0) / 1e6;

    System.out.printf(Locale.ROOT, "SABR calibration (adjoint-gradient L-BFGS)%n%n");
    System.out.printf(Locale.ROOT, "  target params : alpha=%.4f  rho=%+.4f  nu=%.4f%n", trueAlpha, trueRho, trueNu);
    System.out.printf(Locale.ROOT, "  recovered     : alpha=%.4f  rho=%+.4f  nu=%.4f%n",
        r.parameters().get("alpha"), r.parameters().get("rho"), r.parameters().get("nu"));
    System.out.printf(Locale.ROOT, "  residual SSE  : %.3e%n", r.objective());
    System.out.printf(Locale.ROOT, "  iterations    : %d   converged=%s   %.1f ms%n",
        r.iterations(), r.converged(), ms);

    System.out.printf(Locale.ROOT, "%n  strike   target vol   fitted vol%n");
    for (double k : STRIKES) {
      double fit = SabrHagan.blackVol(r.parameters().get("alpha"), BETA,
          r.parameters().get("rho"), r.parameters().get("nu"), F, k, T);
      System.out.printf(Locale.ROOT, "  %.3f    %.6f     %.6f%n",
          k, SabrHagan.blackVol(trueAlpha, BETA, trueRho, trueNu, F, k, T), fit);
    }
  }
}
