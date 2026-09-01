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

import com.nablatensor.reg.frtb.drc.DefaultRiskCharge;
import com.nablatensor.reg.frtb.drc.DefaultRiskPosition;
import com.nablatensor.reg.frtb.drc.DrcParameters.CreditQuality;
import com.nablatensor.reg.frtb.drc.DrcParameters.DrcBucket;
import com.nablatensor.reg.frtb.drc.DrcParameters.Seniority;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * LESSON 17 — DRC offsetting and the <b>hedge-benefit ratio (HBR)</b>. Longs and
 * shorts to different obligors in the same bucket only <em>partly</em> offset
 * (MAR22.23):
 *
 * <pre>
 *   HBR_b = SUM netJTD_long / ( SUM netJTD_long + SUM |netJTD_short| )
 *
 *   DRC_b = max( SUM RW_i * netJTD_long_i  -  HBR_b * SUM RW_i * |netJTD_short_i| , 0 )
 *
 *   DRC   = SUM_b DRC_b        (corporates + sovereigns + local gov, no cross-netting)
 * </pre>
 *
 * Even a notional-for-notional short in the same rating leaves capital, because a
 * balanced book has {@code HBR = 0.5} — at most half the short's risk weight is
 * credited back.
 */
class Lesson17_DrcHedgeBenefitRatioTest {

  private static DefaultRiskPosition corp(String name, CreditQuality q, double notional) {
    return new DefaultRiskPosition(name, DrcBucket.CORPORATES, Seniority.SENIOR, q, notional, notional, 5);
  }

  @Test
  void balancedSameRatingHedge_leavesHalfTheCharge() {
    // A long 1000 BBB, B short 1000 BBB : JTD = +250 / -250 ; HBR = 250 / 500 = 0.5
    // DRC = max( 0.06*250 - 0.5 * 0.06*250 , 0 ) = 15 - 7.5 = 7.5
    List<DefaultRiskPosition> book = List.of(corp("A", CreditQuality.BBB, 1000), corp("B", CreditQuality.BBB, -1000));
    DefaultRiskCharge.Result r = DefaultRiskCharge.of(book).compute();

    assertEquals(0.5, r.hedgeBenefitRatio().get(DrcBucket.CORPORATES), 1e-9);
    assertEquals(7.5, r.perBucket().get(DrcBucket.CORPORATES), 1e-9);
  }

  @Test
  void ratingMismatchMakesTheHedgeWorse() {
    // long BBB (RW 6%), short a *riskier* BB name (RW 15%) of larger notional
    List<DefaultRiskPosition> book = List.of(corp("A", CreditQuality.BBB, 3000), corp("B", CreditQuality.BB, -800));
    DefaultRiskCharge.Result r = DefaultRiskCharge.of(book).compute();

    double jtdLong = 0.25 * 3000;                 // 750
    double jtdShortAbs = 0.25 * 800;              // 200
    double hbr = jtdLong / (jtdLong + jtdShortAbs);
    double expected = Math.max(0.06 * jtdLong - hbr * 0.15 * jtdShortAbs, 0.0);

    assertEquals(hbr, r.hedgeBenefitRatio().get(DrcBucket.CORPORATES), 1e-9);
    assertEquals(expected, r.perBucket().get(DrcBucket.CORPORATES), 1e-9);
  }

  @Test
  void bucketsAreSummedWithNoDiversification() {
    List<DefaultRiskPosition> book = List.of(
        corp("A", CreditQuality.BBB, 1000),
        new DefaultRiskPosition("SOV", DrcBucket.SOVEREIGNS, Seniority.SENIOR, CreditQuality.AA, 5000, 5000, 10));
    DefaultRiskCharge.Result r = DefaultRiskCharge.of(book).compute();

    double drcCorp = 0.06 * 0.25 * 1000;          // long only -> HBR 1, no short -> just RW * JTD
    double drcSov = 0.02 * 0.25 * 5000;
    assertEquals(drcCorp + drcSov, r.total(), 1e-9);
    assertTrue(r.total() > Math.max(drcCorp, drcSov), "no netting across buckets: it is a plain sum");
  }
}
