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
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.reg.frtb.sbm.SbmCharge;
import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * LESSON 4 — the {@code max(0, ...)} under every square root (MAR21.4.4 and the
 * "alternative specification", MAR21.4.7).
 *
 * <p>The correlations MAR21 prescribes are not guaranteed to form a positive
 * semi-definite matrix — especially in the LOW scenario — so the sum inside a
 * square root can go negative. The framework simply <b>floors it at zero</b>:
 *
 * <pre>
 *   K_b = sqrt( max( 0 , SUM WS^2 + SUM rho*WS*WS ) )
 *   S_b = clamp( SUM WS , -K_b , +K_b )
 * </pre>
 *
 * So a strongly-negative correlation can drive a bucket charge all the way to 0,
 * but never to an imaginary number, and never below 0.
 */
class Lesson04_AlternativeSpecificationFloorTest {

  @Test
  void stronglyNegativeCorrelation_floorsTheBucketChargeAtZero() {
    // three equal weighted sensitivities, rho = -0.9 between every pair
    Sensitivities book = Sensitivities.builder()
        .add(RiskFactor.equityDelta("1", "A"), 1.0)
        .add(RiskFactor.equityDelta("1", "B"), 1.0)
        .add(RiskFactor.equityDelta("1", "C"), 1.0)
        .build();

    // inner sum = 3*(1)^2 + 6 ordered pairs * (-0.9) * 1 * 1 = 3 - 5.4 = -2.4  ->  max(0,-2.4) = 0
    double medium = SbmCharge.of(TourProfiles.flat(1.0, -0.9, 0.0))
        .compute(book, List.of())
        .perScenario().get(CorrelationScenario.MEDIUM);

    assertEquals(0.0, medium, 1e-12);
  }

  @Test
  void theChargeIsNeverNegative_underAnyScenario() {
    Sensitivities book = Sensitivities.builder()
        .add(RiskFactor.equityDelta("1", "A"), 5.0)
        .add(RiskFactor.equityDelta("1", "B"), -5.0)
        .add(RiskFactor.equityDelta("2", "C"), 7.0)
        .build();

    SbmCharge.Result r = SbmCharge.of(TourProfiles.flat(1.0, -0.8, -0.8)).compute(book, List.of());
    for (CorrelationScenario sc : CorrelationScenario.values()) {
      assertTrue(r.perScenario().get(sc) >= 0.0, sc + " charge is non-negative");
    }
  }
}
