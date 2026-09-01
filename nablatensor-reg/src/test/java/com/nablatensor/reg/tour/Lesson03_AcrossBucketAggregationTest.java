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
 * LESSON 3 — the second level: aggregating the <b>bucket totals</b>.
 *
 * <pre>
 *   S_b    = clamp( SUM_k WS_k , -K_b , +K_b )        (a signed, capped bucket total)
 *   charge = sqrt( SUM_b K_b^2  +  SUM_{b != c} gamma_bc * S_b * S_c )
 * </pre>
 *
 * <pre>
 *   bucket 1 ─ K_1, S_1 ─┐
 *                        ├─►  sqrt( K_1^2 + K_2^2 + 2*gamma*S_1*S_2 )
 *   bucket 2 ─ K_2, S_2 ─┘
 * </pre>
 *
 * {@code gamma} is the cross-bucket correlation. {@code S_b} is clamped to
 * {@code [-K_b, K_b]} so a bucket can never contribute more (in magnitude) at the
 * top level than its own standalone charge.
 */
class Lesson03_AcrossBucketAggregationTest {

  // one factor per bucket, so K_b = |WS_b| and S_b = WS_b
  private static final Sensitivities BOOK = Sensitivities.builder()
      .add(RiskFactor.equityDelta("1", "A"), 3.0)   // WS_1 = 3  (rw = 1)
      .add(RiskFactor.equityDelta("2", "B"), 4.0)   // WS_2 = 4
      .build();

  private static double mediumCharge(double gamma) {
    return SbmCharge.of(TourProfiles.flat(1.0, 0.0, gamma))
        .compute(BOOK, List.of())
        .perScenario().get(CorrelationScenario.MEDIUM);
  }

  @Test
  void gammaZero_bucketsCombineInQuadrature() {
    assertEquals(5.0, mediumCharge(0.0), 1e-9);       // sqrt(9 + 16)
  }

  @Test
  void gammaOne_bucketsAddLinearly() {
    assertEquals(7.0, mediumCharge(1.0), 1e-9);       // sqrt(9 + 16 + 2*1*12)
  }

  @Test
  void negativeGamma_canOnlyHelpDownToTheFloor() {
    // sqrt(9 + 16 + 2*(-1)*12) = sqrt(1)
    assertEquals(1.0, mediumCharge(-1.0), 1e-9);
  }
}
