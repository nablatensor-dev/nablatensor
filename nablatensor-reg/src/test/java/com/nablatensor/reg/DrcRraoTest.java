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
package com.nablatensor.reg;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.nablatensor.reg.frtb.drc.DefaultRiskCharge;
import com.nablatensor.reg.frtb.drc.DefaultRiskPosition;
import com.nablatensor.reg.frtb.drc.DrcParameters.CreditQuality;
import com.nablatensor.reg.frtb.drc.DrcParameters.DrcBucket;
import com.nablatensor.reg.frtb.drc.DrcParameters.Seniority;
import com.nablatensor.reg.frtb.drc.Jtd;
import com.nablatensor.reg.frtb.rrao.Rrao;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Default Risk Charge and Residual Risk Add-On, reconciled to hand-worked figures. */
class DrcRraoTest {

  @Test
  void grossJtdMatchesTheFormula() {
    // senior (LGD 25%), long 3000 at par, 5y
    DefaultRiskPosition longBond = new DefaultRiskPosition(
        "CORP_A", DrcBucket.CORPORATES, Seniority.SENIOR, CreditQuality.BBB, 3000, 3000, 5);
    assertEquals(0.25 * 3000 + (3000 - 3000), Jtd.gross(longBond), 1e-9);

    // short 800, market value -790, 3y -> no sub-year scaling
    DefaultRiskPosition shortBond = new DefaultRiskPosition(
        "CORP_B", DrcBucket.CORPORATES, Seniority.SENIOR, CreditQuality.BB, -800, -790, 3);
    assertEquals(0.25 * -800 + (-790 - -800), Jtd.gross(shortBond), 1e-9);

    // 6-month position scales by 0.5
    DefaultRiskPosition shortDated = new DefaultRiskPosition(
        "CORP_C", DrcBucket.CORPORATES, Seniority.SENIOR, CreditQuality.A, 1000, 1000, 0.5);
    assertEquals((0.25 * 1000) * 0.5, Jtd.gross(shortDated), 1e-9);
  }

  @Test
  void drcNonSecReconcilesToAHandWorkedExample() {
    List<DefaultRiskPosition> book = List.of(
        new DefaultRiskPosition("CORP_A", DrcBucket.CORPORATES, Seniority.SENIOR, CreditQuality.BBB, 3000, 3000, 5),
        new DefaultRiskPosition("CORP_B", DrcBucket.CORPORATES, Seniority.SENIOR, CreditQuality.BB, -800, -790, 3),
        new DefaultRiskPosition("SOV_X", DrcBucket.SOVEREIGNS, Seniority.SENIOR, CreditQuality.AA, 5000, 5000, 10));

    // ---- fresh reference ----
    double jtdA = 0.25 * 3000;                       // 750  (long)
    double jtdB = 0.25 * -800 + (-790 - -800);       // -190 (short)
    double jtdSov = 0.25 * 5000;                     // 1250 (long)

    double sumLong = jtdA;
    double sumShortAbs = -jtdB;
    double hbrCorp = sumLong / (sumLong + sumShortAbs);
    double wLongCorp = CreditQuality.BBB.riskWeight() * jtdA;
    double wShortCorp = CreditQuality.BB.riskWeight() * -jtdB;
    double drcCorp = Math.max(wLongCorp - hbrCorp * wShortCorp, 0.0);

    double drcSov = Math.max(CreditQuality.AA.riskWeight() * jtdSov - 1.0 * 0.0, 0.0);
    double expected = drcCorp + drcSov;

    DefaultRiskCharge.Result r = DefaultRiskCharge.of(book).compute();
    assertEquals(expected, r.total(), 1e-9);
    assertEquals(drcCorp, r.perBucket().get(DrcBucket.CORPORATES), 1e-9);
    assertEquals(drcSov, r.perBucket().get(DrcBucket.SOVEREIGNS), 1e-9);
    assertEquals(hbrCorp, r.hedgeBenefitRatio().get(DrcBucket.CORPORATES), 1e-9);
  }

  @Test
  void rraoIsAWeightedGrossNotionalSum() {
    List<Rrao.ResidualRiskPosition> book = List.of(
        new Rrao.ResidualRiskPosition("longevity-swap", 1_000_000, Rrao.ResidualRiskKind.EXOTIC_UNDERLYING),
        new Rrao.ResidualRiskPosition("digital-1", 2_000_000, Rrao.ResidualRiskKind.GAP_RISK),
        new Rrao.ResidualRiskPosition("callable-1", 500_000, Rrao.ResidualRiskKind.BEHAVIOURAL_RISK));

    Rrao.Result r = Rrao.of(book).compute();
    assertEquals(0.010 * 1_000_000, r.exotic(), 1e-9);
    assertEquals(0.001 * 2_000_000 + 0.001 * 500_000, r.other(), 1e-9);
    assertEquals(10_000 + 2_500, r.total(), 1e-9);
  }
}
