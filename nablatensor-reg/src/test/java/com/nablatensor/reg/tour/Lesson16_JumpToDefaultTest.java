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

import com.nablatensor.reg.frtb.drc.DefaultRiskPosition;
import com.nablatensor.reg.frtb.drc.DrcParameters.CreditQuality;
import com.nablatensor.reg.frtb.drc.DrcParameters.DrcBucket;
import com.nablatensor.reg.frtb.drc.DrcParameters.Seniority;
import com.nablatensor.reg.frtb.drc.Jtd;
import org.junit.jupiter.api.Test;

/**
 * LESSON 16 — the Default Risk Charge starts from <b>jump-to-default (JTD)</b>:
 * the loss if the issuer defaults right now (MAR22.8):
 *
 * <pre>
 *   gross JTD = ( LGD * notional + (marketValue - notional) ) * maturityScale
 *
 *   LGD by seniority:  senior 25% | non-senior 75% | covered bond 25% | equity 100%
 *   maturityScale   :  1                        for equity (1-year floor)
 *                      clamp(maturityYears, 0.25, 1)   otherwise
 * </pre>
 *
 * A long position has a positive JTD (you lose on default), a short has a
 * negative JTD (you gain).
 */
class Lesson16_JumpToDefaultTest {

  private static DefaultRiskPosition pos(Seniority sen, double notional, double mv, double mat) {
    return new DefaultRiskPosition("X", DrcBucket.CORPORATES, sen, CreditQuality.BBB, notional, mv, mat);
  }

  @Test
  void lgdDependsOnSeniority() {
    assertEquals(0.25 * 1000, Jtd.gross(pos(Seniority.SENIOR, 1000, 1000, 5)), 1e-9);
    assertEquals(0.75 * 1000, Jtd.gross(pos(Seniority.NON_SENIOR, 1000, 1000, 5)), 1e-9);
    assertEquals(1.00 * 1000, Jtd.gross(pos(Seniority.EQUITY, 1000, 1000, 5)), 1e-9);
  }

  @Test
  void markToMarketGainOrLossIsFoldedIn() {
    // a senior bond marked at 970 (already down 30): JTD = 0.25*1000 + (970 - 1000) = 220
    assertEquals(0.25 * 1000 + (970 - 1000), Jtd.gross(pos(Seniority.SENIOR, 1000, 970, 5)), 1e-9);
  }

  @Test
  void shortDatedPositionsAreScaledDown_withAThreeMonthFloor() {
    double sixMonth = Jtd.gross(pos(Seniority.SENIOR, 1000, 1000, 0.5));
    double oneMonth = Jtd.gross(pos(Seniority.SENIOR, 1000, 1000, 1.0 / 12.0));
    assertEquals(250 * 0.5, sixMonth, 1e-9);
    assertEquals(250 * 0.25, oneMonth, 1e-9);   // 1/12 < 0.25 -> floored at 0.25
  }

  @Test
  void aShortPositionHasNegativeJtd() {
    assertEquals(-0.25 * 1000, Jtd.gross(pos(Seniority.SENIOR, -1000, -1000, 5)), 1e-9);
  }
}
