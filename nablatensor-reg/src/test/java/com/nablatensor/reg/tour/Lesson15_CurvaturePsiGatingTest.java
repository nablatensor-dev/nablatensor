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
package com.nablatensor.reg.tour;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.nablatensor.reg.frtb.sbm.CurvatureRepricing;
import com.nablatensor.reg.frtb.sbm.SbmCharge;
import com.nablatensor.reg.frtb.sbm.equity.EquitySbmProfile;
import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * LESSON 15 — the curvature-only {@code psi} gate (MAR21.5.4). Curvature
 * aggregation differs from delta/vega in three ways: the diagonal term is
 * {@code max(CVR, 0)^2}, the correlations are <b>squared</b>, and cross terms
 * are switched off when both legs are curvature <em>benefits</em>:
 *
 * <pre>
 *   psi(CVR_k, CVR_l) = 0   if CVR_k < 0 AND CVR_l < 0
 *                     = 1   otherwise
 * </pre>
 *
 * Without the gate, two long options (both {@code CVR < 0}) could manufacture a
 * spurious positive cross term. With it, they simply drop out.
 */
class Lesson15_CurvaturePsiGatingTest {

  private static final Sensitivities NO_DELTA = Sensitivities.empty();

  /** A repricing engineered (delta = 0, pvUp = pvDown = -target) to yield exactly {@code CVR = target}. */
  private static CurvatureRepricing withCvr(String name, double target) {
    return new CurvatureRepricing(RiskFactor.equityDelta("5", name), 1.0, 0.0, 0.0, -target, -target);
  }

  private static double curvatureCharge(CurvatureRepricing... crs) {
    return SbmCharge.of(EquitySbmProfile.INSTANCE)
        .compute(NO_DELTA, List.of(crs))
        .perScenario().get(CorrelationScenario.MEDIUM);
  }

  @Test
  void bothLegsAreBenefits_theCrossTermIsGatedOffAndTheChargeIsZero() {
    // CVR = (-4, -9) : both negative -> diagonal is max(.,0)^2 = 0, psi = 0 -> K_b = 0
    assertEquals(0.0, curvatureCharge(withCvr("A", -4.0), withCvr("B", -9.0)), 1e-12);
  }

  @Test
  void oneLegIsACharge_theCrossTermIsLive() {
    // CVR = (+4, -9) : psi = 1, rho(A,B) in bucket 5 = 0.25, curvature uses rho^2.
    // The MAR21 sum is over ORDERED pairs k != l, so the cross term appears twice:
    //   K_b = sqrt( max(4,0)^2 + max(-9,0)^2  +  2 * 0.25^2 * 4 * (-9) )
    //       = sqrt( 16 + 0 - 4.5 ) = sqrt(11.5)
    assertEquals(Math.sqrt(11.5), curvatureCharge(withCvr("A", 4.0), withCvr("B", -9.0)), 1e-9);
  }
}
