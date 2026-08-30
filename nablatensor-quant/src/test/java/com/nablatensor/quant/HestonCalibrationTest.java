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

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.SDouble;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Heston calibration against Monte-Carlo target prices: the residual vector is
 * {@code E[payoff_k] - target_k}, its Jacobian comes from one {@link MultiOutput}
 * evaluation per Levenberg-Marquardt step (adjoint through the MC), and common
 * random numbers keep the residual surface smooth enough to recover the
 * parameters exactly.
 */
class HestonCalibrationTest {

  private static final double S0 = 100, R = 0.02, T = 1.0;
  private static final int STEPS = 24;
  private static final long SCEN = 60_000L;
  private static final long SEED = 8675309L;
  private static final double KAPPA = 1.5, THETA = 0.045;      // held fixed
  private static final double[] STRIKES = {90, 100, 110};

  private static final double TRUE_V0 = 0.05;
  private static final double TRUE_XI = 0.55;
  private static final double TRUE_RHO = -0.6;

  @Test
  void recoversHestonV0XiRhoFromMcTargets() {
    // 1) generate MC target prices at the true parameters, same seed
    double[] target = new double[STRIKES.length];
    try (MultiOutput mo = MultiOutput.of(rec -> measures(rec, TRUE_V0, TRUE_XI, TRUE_RHO)).on("cpu-jit").build()) {
      MultiOutput.Result r = mo.run(SCEN, SEED);
      for (int i = 0; i < STRIKES.length; i++) {
        target[i] = r.value("k" + i);
        assertTrue(target[i] > 0, "target price positive");
      }
    }

    // 2) calibrate (v0, xi, rho) from a perturbed start
    final double[] tgt = target;
    Calibrator.Result res = Calibrator.leastSquares(rec -> {
          SDouble v0 = rec.input("v0", 0.03);
          SDouble xi = rec.input("xi", 0.35);
          SDouble rho = rec.input("rho", -0.2);
          Map<String, SDouble> m = hestonMeasures(rec, v0, xi, rho);
          Map<String, SDouble> residuals = new LinkedHashMap<>();
          for (int i = 0; i < STRIKES.length; i++) {
            residuals.put("k" + i, m.get("k" + i).sub(tgt[i]));
          }
          return residuals;
        })
        .parameter("v0", 0.03, 1e-3, 0.5)
        .parameter("xi", 0.35, 1e-2, 3.0)
        .parameter("rho", -0.2, -0.98, 0.5)
        .scenarios(SCEN)
        .seed(SEED)
        .maxIterations(40)
        .tolerance(1e-10)
        .solve();

    assertTrue(res.objective() < 1e-6, "residual SSE small: " + res.objective());
    assertEquals(TRUE_V0, res.parameters().get("v0"), 2e-3, "v0 recovered");
    assertEquals(TRUE_XI, res.parameters().get("xi"), 3e-2, "xi recovered");
    assertEquals(TRUE_RHO, res.parameters().get("rho"), 3e-2, "rho recovered");
  }

  // ---- helpers -----------------------------------------------------------

  private static Map<String, SDouble> measures(AadRecorder rec, double v0, double xi, double rho) {
    return hestonMeasures(rec, rec.constant(v0), rec.constant(xi), rec.constant(rho));
  }

  private static Map<String, SDouble> hestonMeasures(AadRecorder rec, SDouble v0, SDouble xi, SDouble rho) {
    SDouble[] terminal = hestonTerminal(rec, v0, xi, rho);
    SDouble disc = rec.constant(Math.exp(-R * T));
    Map<String, SDouble> m = new LinkedHashMap<>();
    for (int i = 0; i < STRIKES.length; i++) {
      m.put("k" + i, terminal[0].sub(STRIKES[i]).max(0.0).mul(disc));
    }
    return m;
  }

  /** Full-truncation Euler Heston to T; returns {terminal spot}. */
  private static SDouble[] hestonTerminal(AadRecorder rec, SDouble v0, SDouble xi, SDouble rho) {
    double dt = T / STEPS, sqrtDt = Math.sqrt(dt);
    SDouble rhoBar = rho.mul(rho).neg().add(1.0).sqrt();
    SDouble s = rec.constant(S0);
    SDouble v = v0;
    for (int t = 0; t < STEPS; t++) {
      SDouble z1 = rec.randn();
      SDouble z2 = rho.mul(z1).add(rhoBar.mul(rec.randn()));
      SDouble vPlus = v.max(0.0);
      SDouble sqrtV = vPlus.sqrt();
      v = v.add(rec.constant(KAPPA).mul(rec.constant(THETA).sub(vPlus)).mul(dt))
          .add(xi.mul(sqrtV).mul(sqrtDt).mul(z2));
      s = s.mul(rec.constant(R).sub(vPlus.mul(0.5)).mul(dt).add(sqrtV.mul(sqrtDt).mul(z1)).exp());
    }
    return new SDouble[] {s};
  }
}
