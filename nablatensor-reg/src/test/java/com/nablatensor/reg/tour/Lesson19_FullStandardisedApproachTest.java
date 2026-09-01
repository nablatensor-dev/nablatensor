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

import com.nablatensor.reg.frtb.drc.DefaultRiskPosition;
import com.nablatensor.reg.frtb.drc.DrcParameters.CreditQuality;
import com.nablatensor.reg.frtb.drc.DrcParameters.DrcBucket;
import com.nablatensor.reg.frtb.drc.DrcParameters.Seniority;
import com.nablatensor.reg.frtb.rrao.Rrao;
import com.nablatensor.reg.frtb.sa.FrtbSa;
import com.nablatensor.reg.frtb.sbm.SbmCharge;
import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * LESSON 19 — the whole thing. {@link FrtbSa} runs SBM for every populated risk
 * class, then adds DRC and RRAO:
 *
 * <pre>
 *   FRTB SA own-funds requirement
 *       = SUM over risk classes of  max( HIGH, MEDIUM, LOW )      <- SBM
 *       + DRC                                                     <- jump-to-default
 *       + RRAO                                                    <- gross-notional backstop
 *
 *   no diversification between the three blocks — they are simply summed.
 * </pre>
 */
class Lesson19_FullStandardisedApproachTest {

  private static final Sensitivities BOOK = Sensitivities.builder()
      .add(RiskFactor.equityDelta("5", "ACME"), 1_000.0)
      .add(RiskFactor.equityDelta("5", "GLOBEX"), -400.0)
      .add(RiskFactor.girrDelta("EUR", "OIS", 2), 9_000.0)
      .add(RiskFactor.girrDelta("EUR", "OIS", 10), -3_000.0)
      .add(RiskFactor.fxDelta("EURUSD"), 750_000.0)
      .build();

  private static final List<DefaultRiskPosition> DRC = List.of(
      new DefaultRiskPosition("CORP_A", DrcBucket.CORPORATES, Seniority.SENIOR, CreditQuality.BBB, 3000, 3000, 5),
      new DefaultRiskPosition("CORP_B", DrcBucket.CORPORATES, Seniority.SENIOR, CreditQuality.BB, -800, -790, 3));

  private static final List<Rrao.ResidualRiskPosition> RRAO = List.of(
      new Rrao.ResidualRiskPosition("longevity", 1_000_000, Rrao.ResidualRiskKind.EXOTIC_UNDERLYING));

  @Test
  void totalIsSbmPlusDrcPlusRrao() {
    FrtbSa.Result r = FrtbSa.of("EUR").sbm(BOOK, List.of()).drc(DRC).rrao(RRAO).compute();

    double sbmFromClasses = r.perRiskClass().values().stream().mapToDouble(SbmCharge.Result::total).sum();
    assertEquals(sbmFromClasses, r.sbm(), 1e-9);
    assertEquals(r.sbm() + r.drc() + r.rrao(), r.total(), 1e-9);

    assertTrue(r.perRiskClass().keySet().containsAll(List.of(RiskClass.EQUITY, RiskClass.GIRR, RiskClass.FX)));
    assertTrue(r.drc() > 0.0);
    assertEquals(0.01 * 1_000_000, r.rrao(), 1e-9);
  }

  @Test
  void theCorepViewMirrorsTheResult() {
    FrtbSa.Result r = FrtbSa.of("EUR").sbm(BOOK, List.of()).drc(DRC).rrao(RRAO).compute();

    assertEquals(r.total(), r.corep().total(), 1e-12);
    assertEquals(r.sbm(), r.corep().sbmTotal(), 1e-12);
    assertEquals("EUR", r.corep().reportingCurrency());
    assertEquals("Basel MAR21", r.corep().parameterSet());
    r.corep().perRiskClass().forEach((rc, row) ->
        assertEquals(row.delta() + row.vega() + row.curvature(), row.classCharge(), 1e-9, rc + " row adds up"));
  }

  @Test
  void anEmptyBookCostsNothing() {
    assertEquals(0.0, FrtbSa.of("EUR").compute().total(), 0.0);
  }
}
