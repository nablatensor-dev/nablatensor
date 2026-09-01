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

import com.nablatensor.reg.frtb.sbm.SbmCharge;
import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * LESSON 5 — the three correlation scenarios (MAR21.6). Every prescribed
 * correlation is a MEDIUM value; the bank recomputes the whole charge with the
 * correlations shifted up and down, and <b>keeps the largest of the three</b>:
 *
 * <pre>
 *   HIGH   : rho -> min(1.25 * rho, 1)          correlations pushed toward 1
 *   MEDIUM : rho                                 as prescribed
 *   LOW    : rho -> max(2*rho - 1, 0.75 * rho)   correlations pushed toward 0 / negative
 * </pre>
 *
 * Which one bites depends on the book:
 *
 * <pre>
 *   long + long   : more correlation hurts  ->  HIGH usually binds
 *   long + short  : the hedge looks worse when things decorrelate  ->  LOW usually binds
 * </pre>
 */
class Lesson05_ThreeCorrelationScenariosTest {

  @Test
  void scenarioTransforms_matchTheFormula() {
    assertEquals(0.625, CorrelationScenario.HIGH.apply(0.5), 1e-12);            // 1.25 * 0.5
    assertEquals(1.0, CorrelationScenario.HIGH.apply(0.9), 1e-12);             // capped at 1
    assertEquals(0.5, CorrelationScenario.MEDIUM.apply(0.5), 1e-12);
    assertEquals(0.375, CorrelationScenario.LOW.apply(0.5), 1e-12);            // max(2*0.5-1, 0.75*0.5) = max(0, 0.375)
    assertEquals(0.225, CorrelationScenario.LOW.apply(0.3), 1e-12);            // max(-0.4, 0.225) : the 0.75*rho branch wins
    assertEquals(0.8, CorrelationScenario.LOW.apply(0.9), 1e-12);             // max(0.8, 0.675) : the 2*rho-1 branch wins for high rho
  }

  @Test
  void longLongBook_isBoundByTheHighScenario() {
    Sensitivities longLong = Sensitivities.builder()
        .add(RiskFactor.equityDelta("1", "A"), 4.0)
        .add(RiskFactor.equityDelta("1", "B"), 3.0)
        .build();

    SbmCharge.Result r = SbmCharge.of(TourProfiles.flat(1.0, 0.5, 0.5)).compute(longLong, List.of());

    // HIGH:  sqrt(25 + 2*0.625*12) = sqrt(40)
    // MED :  sqrt(25 + 2*0.5  *12) = sqrt(37)
    // LOW :  sqrt(25 + 2*0.375*12) = sqrt(34)
    assertEquals(CorrelationScenario.HIGH, r.bindingScenario());
    assertEquals(Math.sqrt(40.0), r.total(), 1e-9);
  }

  @Test
  void longShortBook_isBoundByTheLowScenario() {
    Sensitivities longShort = Sensitivities.builder()
        .add(RiskFactor.equityDelta("1", "A"), 4.0)
        .add(RiskFactor.equityDelta("1", "B"), -3.0)
        .build();

    SbmCharge.Result r = SbmCharge.of(TourProfiles.flat(1.0, 0.5, 0.5)).compute(longShort, List.of());

    // HIGH:  sqrt(25 - 2*0.625*12) = sqrt(10)
    // MED :  sqrt(25 - 2*0.5  *12) = sqrt(13)
    // LOW :  sqrt(25 - 2*0.375*12) = sqrt(16) = 4
    assertEquals(CorrelationScenario.LOW, r.bindingScenario());
    assertEquals(4.0, r.total(), 1e-9);
  }
}
