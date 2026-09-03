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
package com.nablatensor.cva;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class HazardCurveTest {

  @Test
  void survivalStartsAtOneAndDecreases() {
    HazardCurve curve = HazardCurve.fromFlatSpread(120.0, 0.4, 10.0);
    assertEquals(1.0, curve.survival(0.0), 1.0e-12);
    double previous = 1.0;
    for (double t = 0.5; t <= 10.0; t += 0.5) {
      double survival = curve.survival(t);
      assertTrue(survival < previous, "survival must be strictly decreasing at t=" + t);
      assertTrue(survival > 0.0, "survival must stay positive");
      previous = survival;
    }
  }

  @Test
  void flatSpreadGivesTheTextbookHazard() {
    double recovery = 0.4;
    HazardCurve curve = HazardCurve.fromFlatSpread(150.0, recovery, 5.0);
    assertEquals(150.0e-4 / (1.0 - recovery), curve.hazardAt(3.0), 1.0e-12);
  }

  @Test
  void bootstrapReprojectsNearlyFlatQuotes() {
    double recovery = 0.4;
    List<CdsQuote> quotes = List.of(
        new CdsQuote(1.0, 100.0), new CdsQuote(3.0, 100.0),
        new CdsQuote(5.0, 100.0), new CdsQuote(10.0, 100.0));
    HazardCurve curve = HazardCurve.bootstrap(quotes, recovery, t -> Math.exp(-0.03 * t));

    double approximate = 100.0e-4 / (1.0 - recovery);
    for (double hazard : curve.forwardHazards()) {
      assertEquals(approximate, hazard, 0.15 * approximate,
          "flat quotes should bootstrap to a near-flat hazard near s/(1-R)");
    }
  }

  @Test
  void parallelShiftRaisesEveryHazardAndLowersSurvival() {
    HazardCurve base = HazardCurve.fromFlatSpread(100.0, 0.4, 10.0);
    HazardCurve shocked = base.shockedBySpread(10.0, 0.4);
    assertTrue(shocked.hazardAt(2.0) > base.hazardAt(2.0));
    assertTrue(shocked.survival(5.0) < base.survival(5.0));
    assertEquals(10.0e-4 / 0.6, shocked.hazardAt(2.0) - base.hazardAt(2.0), 1.0e-12);
  }
}
