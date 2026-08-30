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
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Levenberg-Marquardt over a recorded residual vector (Jacobian from MultiOutput). */
class CalibratorLmTest {

  private static final double F = 0.05;
  private static final double T = 1.0;
  private static final double BETA = 0.5;
  private static final double[] STRIKES = {0.030, 0.040, 0.045, 0.055, 0.062, 0.070};
  private static final double TRUE_ALPHA = 0.27;
  private static final double TRUE_RHO = -0.28;
  private static final double TRUE_NU = 0.52;

  @Test
  void levenbergMarquardtRecoversTheSabrSmile() {
    double[] target = new double[STRIKES.length];
    for (int i = 0; i < STRIKES.length; i++) {
      target[i] = SabrHagan.blackVol(TRUE_ALPHA, BETA, TRUE_RHO, TRUE_NU, F, STRIKES[i], T);
    }

    Calibrator.Result r = Calibrator.leastSquares(rec -> {
          SDouble alpha = rec.input("alpha", 0.20);
          SDouble rho = rec.input("rho", 0.0);
          SDouble nu = rec.input("nu", 0.30);
          SDouble beta = rec.constant(BETA);
          Map<String, SDouble> res = new LinkedHashMap<>();
          for (int i = 0; i < STRIKES.length; i++) {
            res.put("k" + i,
                SabrHagan.blackVol(rec, alpha, beta, rho, nu, F, STRIKES[i], T).sub(target[i]));
          }
          return res;
        })
        .parameter("alpha", 0.20, 1e-4, 2.0)
        .parameter("rho", 0.0, -0.999, 0.999)
        .parameter("nu", 0.30, 1e-4, 5.0)
        .maxIterations(60)
        .tolerance(1e-12)
        .solve();

    assertTrue(r.converged() || r.objective() < 1e-12, "LM converged, obj=" + r.objective());
    assertEquals(TRUE_ALPHA, r.parameters().get("alpha"), 2e-3, "alpha");
    assertEquals(TRUE_RHO, r.parameters().get("rho"), 5e-3, "rho");
    assertEquals(TRUE_NU, r.parameters().get("nu"), 5e-3, "nu");
    assertTrue(r.iterations() <= 40, "LM is fast: " + r.iterations() + " iterations");
  }
}
