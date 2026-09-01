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
 * LESSON 2 — aggregating factors <b>inside one bucket</b>:
 *
 * <pre>
 *   K_b = sqrt( SUM_k WS_k^2  +  SUM_{k != l} rho_kl * WS_k * WS_l )
 * </pre>
 *
 * {@code rho} is the regulator's view of how much two factors in the same bucket
 * move together. It interpolates between "independent" and "identical":
 *
 * <pre>
 *   WS = (3, 4)                rho = 0            rho = 1            rho = -1
 *                          (independent)      (identical)        (opposite)
 *        4  ^  .                 |4                 |                  |
 *           | /|                 |                  |7  = 3+4          |
 *           |/ | 5   K_b =       |     K_b = 5      |     K_b = 7     |1 = |3-4|
 *   --------+--+--->  sqrt(9+16) |                  |                 |    K_b = 1
 *           0  3                 +---3---           +---               +---
 * </pre>
 *
 * Higher correlation ⇒ less diversification ⇒ bigger capital.
 */
class Lesson02_WithinBucketCorrelationTest {

  private static final Sensitivities BOOK = Sensitivities.builder()
      .add(RiskFactor.equityDelta("1", "A"), 3.0)
      .add(RiskFactor.equityDelta("1", "B"), 4.0)
      .build();

  private static double mediumCharge(double rho) {
    return SbmCharge.of(TourProfiles.flat(1.0, rho, 0.0))
        .compute(BOOK, List.of())
        .perScenario().get(CorrelationScenario.MEDIUM);
  }

  @Test
  void rhoZero_isPythagoras() {
    assertEquals(5.0, mediumCharge(0.0), 1e-9);       // sqrt(3^2 + 4^2)
  }

  @Test
  void rhoOne_isTheArithmeticSum() {
    assertEquals(7.0, mediumCharge(1.0), 1e-9);       // sqrt(9 + 16 + 2*1*12) = |3 + 4|
  }

  @Test
  void rhoMinusOne_isTheDifference() {
    assertEquals(1.0, mediumCharge(-1.0), 1e-9);      // sqrt(9 + 16 - 24) = |3 - 4|
  }

  @Test
  void chargeIsMonotoneInCorrelation() {
    assertTrue(mediumCharge(0.0) < mediumCharge(0.5));
    assertTrue(mediumCharge(0.5) < mediumCharge(1.0));
  }
}
