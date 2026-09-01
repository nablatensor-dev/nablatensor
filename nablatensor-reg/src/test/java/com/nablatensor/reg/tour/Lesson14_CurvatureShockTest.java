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
 * LESSON 14 — curvature: the charge for the risk that delta misses on a big
 * move. It needs <b>two full repricings</b> per risk factor (shock up, shock
 * down) and strips the linear (delta) part out of each (MAR21.5):
 *
 * <pre>
 *   shock = RW_curv * level                (here: 0.30 * 100 = 30)
 *   up   = PV(spot + shock) - PV(spot) - shock * delta
 *   down = PV(spot - shock) - PV(spot) + shock * delta
 *   CVR  = - min(up, down)                 (the worse of the two, sign-flipped)
 *
 *          PV
 *           |        short option (concave): both shocks lose vs the delta line
 *           |     __________            CVR > 0  -> a charge
 *           |    /          \
 *   --------+---/------+------\--------- delta line (linear approx)
 *           |  /       |       \
 *          -30       spot      +30
 * </pre>
 *
 * A long option is convex, so both bracket terms are positive, {@code CVR} is
 * negative, and — floored by {@code max(CVR, 0)} in the aggregation — its
 * isolated curvature charge is 0.
 */
class Lesson14_CurvatureShockTest {

  private static final Sensitivities NO_DELTA = Sensitivities.empty();

  private static double curvatureCharge(CurvatureRepricing cr) {
    return SbmCharge.of(EquitySbmProfile.INSTANCE)
        .compute(NO_DELTA, List.of(cr))
        .perScenario().get(CorrelationScenario.MEDIUM);
  }

  @Test
  void shortOption_hasAPositiveCurvatureCharge() {
    // short a call: delta = -0.6 ; PV(100) = -10, PV(130) = -32, PV(70) = -1
    CurvatureRepricing shortCall = new CurvatureRepricing(
        RiskFactor.equityDelta("5", "ACME"), 100.0, -0.6, -10.0, -32.0, -1.0);

    // shock = 0.30 * 100 = 30
    // up   = -32 - (-10) - 30*(-0.6) = -4
    // down = -1  - (-10) + 30*(-0.6) = -9
    // CVR  = -min(-4, -9) = 9
    assertEquals(9.0, curvatureCharge(shortCall), 1e-9);
  }

  @Test
  void longOption_hasNoIsolatedCurvatureCharge() {
    // long the same call: delta = +0.6 ; PV(100) = +10, PV(130) = +32, PV(70) = +1
    CurvatureRepricing longCall = new CurvatureRepricing(
        RiskFactor.equityDelta("5", "ACME"), 100.0, 0.6, 10.0, 32.0, 1.0);

    // up = 32 - 10 - 30*0.6 = 4 ; down = 1 - 10 + 30*0.6 = 9 ; CVR = -min(4, 9) = -4
    // -> max(CVR, 0) = 0 in the within-bucket sum
    assertEquals(0.0, curvatureCharge(longCall), 1e-9);
  }
}
