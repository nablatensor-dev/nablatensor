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
 * LESSON 1 — the weighted sensitivity {@code WS_k = RW_k * s_k}.
 *
 * <p>FRTB SA-SBM starts from a book's <b>net sensitivity to each risk factor</b>
 * — exactly what one adjoint (reverse-mode) sweep produces. Each raw sensitivity
 * {@code s_k} is scaled by a prescribed <b>risk weight</b> {@code RW_k}:
 *
 * <pre>
 *   one adjoint sweep            risk weights (MAR21)
 *   ----------------             --------------------
 *   s_1  = dPV/dRF_1  ──┐
 *   s_2  = dPV/dRF_2  ──┤  WS_k = RW_k * s_k   ──►  aggregation (lessons 2, 3)
 *   ...                ──┘
 * </pre>
 *
 * With a single risk factor there is nothing to aggregate: the bucket charge is
 * just {@code |WS|}, and all three correlation scenarios agree.
 */
class Lesson01_WeightedSensitivityTest {

  @Test
  void oneRiskFactor_chargeIsRiskWeightTimesSensitivity() {
    double riskWeight = 0.30;                 // 30% — a stand-in for a MAR21 weight
    double sensitivity = 2_000.0;             // dPV per unit move in the risk factor

    Sensitivities book = Sensitivities.builder()
        .add(RiskFactor.equityDelta("1", "ACME"), sensitivity)
        .build();

    SbmCharge.Result r = SbmCharge.of(TourProfiles.flat(riskWeight, 0.0, 0.0))
        .compute(book, List.of());

    // WS = 0.30 * 2000 = 600 ; with one factor, K_b = |WS| and the charge = K_b
    assertEquals(600.0, r.total(), 1e-9);
    for (CorrelationScenario sc : CorrelationScenario.values()) {
      assertEquals(600.0, r.perScenario().get(sc), 1e-9, sc + " agrees when there is nothing to correlate");
    }
  }

  @Test
  void sign_isCarriedThrough_butTheChargeIsAMagnitude() {
    Sensitivities shortBook = Sensitivities.builder()
        .add(RiskFactor.equityDelta("1", "ACME"), -2_000.0)   // a short position
        .build();

    SbmCharge.Result r = SbmCharge.of(TourProfiles.flat(0.30, 0.0, 0.0)).compute(shortBook, List.of());

    // WS = -600, but K_b = sqrt(WS^2) = 600 : capital does not care about direction here
    assertEquals(600.0, r.total(), 1e-9);
  }
}
