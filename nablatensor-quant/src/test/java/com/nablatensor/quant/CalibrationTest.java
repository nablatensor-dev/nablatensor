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

import com.nablatensor.engine.SDouble;
import org.junit.jupiter.api.Test;

/**
 * Calibration over a recorded objective: fit SABR (alpha, rho, nu) to a Hagan
 * smile generated from a known parameter set. One compiled objective, one
 * adjoint sweep per iteration, box-projected L-BFGS.
 */
class CalibrationTest {

  private static final double F = 0.05;
  private static final double T = 1.0;
  private static final double BETA = 0.5;
  private static final double[] STRIKES = {0.030, 0.040, 0.045, 0.055, 0.060, 0.070};

  private static final double TRUE_ALPHA = 0.28;
  private static final double TRUE_RHO = -0.30;
  private static final double TRUE_NU = 0.55;

  @Test
  void recoversKnownSabrParametersFromASyntheticSmile() {
    double[] targetVol = new double[STRIKES.length];
    for (int i = 0; i < STRIKES.length; i++) {
      targetVol[i] = SabrHagan.blackVol(TRUE_ALPHA, BETA, TRUE_RHO, TRUE_NU, F, STRIKES[i], T);
    }

    Calibrator.Result r = Calibrator.of(rec -> {
          SDouble alpha = rec.input("alpha", 0.20);
          SDouble rho = rec.input("rho", 0.00);
          SDouble nu = rec.input("nu", 0.30);
          SDouble beta = rec.constant(BETA);
          SDouble sse = rec.constant(0.0);
          for (int i = 0; i < STRIKES.length; i++) {
            SDouble model = SabrHagan.blackVol(rec, alpha, beta, rho, nu, F, STRIKES[i], T);
            SDouble d = model.sub(targetVol[i]);
            sse = sse.add(d.mul(d));
          }
          rec.output(sse);
        })
        .parameter("alpha", 0.20, 1e-4, 2.0)
        .parameter("rho", 0.00, -0.999, 0.999)
        .parameter("nu", 0.30, 1e-4, 5.0)
        .maxIterations(80)
        .tolerance(1e-11)
        .solve();

    assertTrue(r.converged(), "L-BFGS converged (grad norm " + r.gradientNorm() + ", " + r.iterations() + " iters)");
    assertTrue(r.objective() < 1e-10, "residual sum of squares driven to zero: " + r.objective());
    assertEquals(TRUE_ALPHA, r.parameters().get("alpha"), 2e-3, "alpha recovered");
    assertEquals(TRUE_RHO, r.parameters().get("rho"), 5e-3, "rho recovered");
    assertEquals(TRUE_NU, r.parameters().get("nu"), 5e-3, "nu recovered");
    assertTrue(r.iterations() <= 60, "recovered in a modest iteration count: " + r.iterations());
  }

  @Test
  void haganTapeFormMatchesTheHostForm() {
    // the differentiable form and the plain-double reference must agree to rounding
    Calibrator.Result r = Calibrator.of(rec -> {
          SDouble alpha = rec.input("alpha", TRUE_ALPHA);
          SDouble rho = rec.input("rho", TRUE_RHO);
          SDouble nu = rec.input("nu", TRUE_NU);
          SDouble beta = rec.constant(BETA);
          SDouble v = SabrHagan.blackVol(rec, alpha, beta, rho, nu, F, 0.06, T);
          rec.output(v);
        })
        .parameter("alpha", TRUE_ALPHA, 1e-4, 2.0)
        .parameter("rho", TRUE_RHO, -0.999, 0.999)
        .parameter("nu", TRUE_NU, 1e-4, 5.0)
        .maxIterations(0)
        .solve();
    double host = SabrHagan.blackVol(TRUE_ALPHA, BETA, TRUE_RHO, TRUE_NU, F, 0.06, T);
    assertEquals(host, r.objective(), 1e-12, "tape vs host Hagan vol");
  }
}
