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
package com.nablatensor.risk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * The two-level correlation aggregation, reconciled to hand-worked arithmetic.
 */
class NestedAggregationTest {

  // bucket 5 (rho .25, RW .30), bucket 6 (rho .25, RW .35), gamma(5,6)=.15
  private static final NestedAggregation.RiskWeight RW = f -> f.bucket().equals("5") ? 0.30 : 0.35;
  private static final NestedAggregation.WithinBucketCorrelation RHO =
      (k, l) -> k.equals(l) ? 1.0 : (k.bucket().equals(l.bucket()) ? 0.25 : 0.0);
  private static final NestedAggregation.AcrossBucketCorrelation GAMMA =
      (b, c) -> b.equals(c) ? 1.0 : 0.15;

  private static RiskFactor eq(String bucket, String name) {
    return RiskFactor.equityDelta(bucket, name);
  }

  @Test
  void deltaAggregationMatchesTheHandCalc() {
    Sensitivities s = Sensitivities.builder()
        .add(eq("5", "A"), 100.0)
        .add(eq("5", "B"), -40.0)
        .add(eq("6", "C"), 50.0)
        .build();

    // WS: A=30, B=-12, C=17.5
    double k5 = Math.sqrt(30 * 30 + 12 * 12 + 2 * 0.25 * 30 * -12);   // sqrt(864)
    double k6 = 17.5;
    double s5 = 18.0;                                                 // clamp(30-12)
    double s6 = 17.5;
    double expected = Math.sqrt(k5 * k5 + k6 * k6 + 2 * 0.15 * s5 * s6);

    NestedAggregation.Result r = NestedAggregation.delta(RW, RHO, GAMMA).aggregate(s);
    assertEquals(k5, r.kb().get("5"), 1e-10, "K_5");
    assertEquals(s5, r.sb().get("5"), 1e-10, "S_5");
    assertEquals(expected, r.total(), 1e-10, "SBM-style total");
  }

  @Test
  void curvatureUsesSquaredCorrelationsAndPsiGating() {
    Sensitivities cvr = Sensitivities.builder()
        .add(eq("5", "A").asCurvature(), 10.0)
        .add(eq("5", "B").asCurvature(), -6.0)
        .add(eq("6", "C").asCurvature(), -3.0)
        .build();

    double k5 = Math.sqrt(100 + 0 + 2 * (0.25 * 0.25) * 10 * -6);     // psi(10,-6)=1
    double k6 = 0.0;                                                  // max(-3,0)^2 = 0
    double s5 = Math.max(-k5, Math.min(10 - 6, k5));                  // 4
    double expected = Math.sqrt(k5 * k5 + k6 * k6);                   // S_6 clamps to 0

    NestedAggregation.Result r = NestedAggregation.curvature(RHO, GAMMA).aggregate(cvr);
    assertEquals(k5, r.kb().get("5"), 1e-10, "K_5 curvature");
    assertEquals(s5, r.sb().get("5"), 1e-10, "S_5 curvature");
    assertEquals(expected, r.total(), 1e-10, "curvature total");
  }

  @Test
  void correlationScenariosTransformAsSpecified() {
    assertEquals(0.25, CorrelationScenario.MEDIUM.apply(0.25), 0);
    assertEquals(0.3125, CorrelationScenario.HIGH.apply(0.25), 1e-12);      // min(1.25*rho, 1)
    assertEquals(0.1875, CorrelationScenario.LOW.apply(0.25), 1e-12);       // max(2*rho-1, .75*rho)
    assertEquals(1.0, CorrelationScenario.HIGH.apply(0.9), 1e-12);          // capped at 1
    assertEquals(0.8, CorrelationScenario.LOW.apply(0.9), 1e-12);           // 2*.9-1
  }

  @Test
  void singleBucketReducesToTheWeightedNorm() {
    Sensitivities s = Sensitivities.builder()
        .add(eq("5", "A"), 10.0).add(eq("5", "B"), 20.0).build();
    // rho=0 override
    NestedAggregation zero = NestedAggregation.delta(f -> 1.0, (k, l) -> k.equals(l) ? 1 : 0, (b, c) -> 0);
    assertEquals(Math.hypot(10, 20), zero.aggregate(s).total(), 1e-12);
    assertTrue(zero.aggregate(Sensitivities.empty()).total() == 0.0);
  }
}
