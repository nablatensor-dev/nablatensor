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

import org.junit.jupiter.api.Test;

/**
 * Feature F1: policy-optimisation LSM produces a Bermudan lower bound near the
 * Longstaff-Schwartz finite-difference reference, a positive early-exercise
 * premium, an American call that does not exercise early, and a sane delta from
 * the same reverse sweep.
 */
class BermudanLsmTest {

  @Test
  void americanPutNearTheLongstaffSchwartzReference() {
    // Longstaff-Schwartz (2001), Table 1: S=K=40, sigma=0.20, T=1, r=0.06
    // finite-difference American put value 2.314.
    EquityMarket m = new EquityMarket(40, 40, 0.20, 0.06, 1.0);
    BermudanLsm.Result r = BermudanLsm.price(m, OptionType.PUT, 20, 6, 3, 0.6, 120_000L, 42L);

    assertTrue(r.price() > r.europeanFloor(), "Bermudan >= European");
    assertTrue(r.earlyExercisePremium() > 0.10, "meaningful early-exercise premium, got " + r.earlyExercisePremium());
    // A lower bound: below the reference by the grid/smoothing gap, not above it by much.
    assertTrue(r.price() > 2.15 && r.price() < 2.36,
        "American put lower bound near 2.314, got " + r.price());
    assertTrue(r.greeks().spot() > -1.0 && r.greeks().spot() < 0.0,
        "put delta in (-1, 0), got " + r.greeks().spot());
    assertTrue(r.standardError() > 0 && r.standardError() < 0.05, "reported standard error sane");
  }

  @Test
  void americanCallOnANonDividendStockDoesNotExerciseEarly() {
    EquityMarket m = new EquityMarket(100, 100, 0.20, 0.05, 1.0);
    BermudanLsm.Result r = BermudanLsm.price(m, OptionType.CALL, 16, 6, 2, 1.0, 100_000L, 7L);
    // Early exercise is never optimal, so the optimiser drives the policy to
    // "hold to expiry" and the price collapses to the European.
    assertEquals(r.europeanFloor(), r.price(), 0.02 * r.europeanFloor(),
        "American call == European call");
    assertTrue(r.greeks().spot() > 0.0 && r.greeks().spot() < 1.0, "call delta in (0, 1)");
  }

  @Test
  void higherVolatilityRaisesTheEarlyExercisePremium() {
    EquityMarket lowVol = new EquityMarket(40, 40, 0.20, 0.06, 1.0);
    EquityMarket highVol = new EquityMarket(40, 40, 0.40, 0.06, 1.0);
    double eepLow = BermudanLsm.price(lowVol, OptionType.PUT, 16, 6, 2, 0.6, 100_000L, 3L)
        .earlyExercisePremium();
    double eepHigh = BermudanLsm.price(highVol, OptionType.PUT, 16, 6, 2, 1.0, 100_000L, 3L)
        .earlyExercisePremium();
    assertTrue(eepHigh > eepLow, "more vol => more early-exercise value: " + eepHigh + " vs " + eepLow);
  }
}
