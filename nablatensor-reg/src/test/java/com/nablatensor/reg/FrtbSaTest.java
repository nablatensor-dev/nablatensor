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
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.reg.frtb.FrtbSaSbm;
import com.nablatensor.reg.frtb.drc.DefaultRiskPosition;
import com.nablatensor.reg.frtb.drc.DrcParameters.CreditQuality;
import com.nablatensor.reg.frtb.drc.DrcParameters.DrcBucket;
import com.nablatensor.reg.frtb.drc.DrcParameters.Seniority;
import com.nablatensor.reg.frtb.rrao.Rrao;
import com.nablatensor.reg.frtb.sa.FrtbSa;
import com.nablatensor.reg.frtb.sbm.CurvatureRepricing;
import com.nablatensor.reg.frtb.sbm.SbmCharge;
import com.nablatensor.reg.frtb.sbm.equity.EquitySbmProfile;
import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The {@link FrtbSa} assembler: the total is {@code SBM + DRC + RRAO} with SBM
 * the per-risk-class max over the three correlation scenarios; the generic
 * engine reproduces the equity-only {@code FrtbSaSbm} facade; the relieved /
 * unrelieved dual runs one book through two parameter sets.
 */
class FrtbSaTest {

  // ---- generic engine == equity facade, for the FrtbSaSbmTest book -----

  private static RiskFactor d(String b, String n) {
    return RiskFactor.equityDelta(b, n);
  }

  private static RiskFactor v(String b, String n, double t) {
    return RiskFactor.equityVega(b, n, t);
  }

  private static final Sensitivities EQ_BOOK = Sensitivities.builder()
      .add(d("5", "ACME"), 1000.0)
      .add(d("5", "GLOBEX"), -400.0)
      .add(d("6", "INITECH"), 800.0)
      .add(v("5", "ACME", 1.0), 50.0)
      .add(v("5", "ACME", 2.0), -20.0)
      .add(v("6", "INITECH", 1.0), 15.0)
      .build();

  private static final List<CurvatureRepricing> EQ_CURV = List.of(
      new CurvatureRepricing(d("5", "ACME"), 100.0, 10.0, 200.0, 260.0, 170.0),
      new CurvatureRepricing(d("6", "INITECH"), 80.0, 6.0, 150.0, 140.0, 180.0));

  @Test
  void genericEngineReproducesTheEquityFacade() {
    List<FrtbSaSbm.CurvatureInput> facadeCurv = List.of(
        new FrtbSaSbm.CurvatureInput(d("5", "ACME"), 100.0, 10.0, 200.0, 260.0, 170.0),
        new FrtbSaSbm.CurvatureInput(d("6", "INITECH"), 80.0, 6.0, 150.0, 140.0, 180.0));
    FrtbSaSbm.Result facade = FrtbSaSbm.equity(EQ_BOOK, facadeCurv);

    SbmCharge.Result generic = SbmCharge.of(EquitySbmProfile.INSTANCE).compute(EQ_BOOK, EQ_CURV);

    assertEquals(facade.delta(), generic.delta(), 1e-9);
    assertEquals(facade.vega(), generic.vega(), 1e-9);
    assertEquals(facade.curvature(), generic.curvature(), 1e-9);
    assertEquals(facade.total(), generic.total(), 1e-9);
    assertEquals(facade.bindingScenario(), generic.bindingScenario());
  }

  // ---- assembler: SBM + DRC + RRAO -----------------------------------

  @Test
  void assemblerSumsSbmDrcAndRrao() {
    Sensitivities book = EQ_BOOK.plus(Sensitivities.builder()
        .add(RiskFactor.girrDelta("EUR", "OIS", 2), 9_000.0)
        .add(RiskFactor.girrDelta("EUR", "OIS", 10), -3_000.0)
        .add(RiskFactor.fxDelta("EURUSD"), 750_000.0)
        .build());

    List<DefaultRiskPosition> drc = List.of(
        new DefaultRiskPosition("CORP_A", DrcBucket.CORPORATES, Seniority.SENIOR, CreditQuality.BBB, 3000, 3000, 5),
        new DefaultRiskPosition("CORP_B", DrcBucket.CORPORATES, Seniority.SENIOR, CreditQuality.BB, -800, -790, 3));
    List<Rrao.ResidualRiskPosition> rrao = List.of(
        new Rrao.ResidualRiskPosition("wx", 1_000_000, Rrao.ResidualRiskKind.EXOTIC_UNDERLYING));

    FrtbSa.Result r = FrtbSa.of("EUR").sbm(book, EQ_CURV).drc(drc).rrao(rrao).compute();

    double sbmFromClasses = r.perRiskClass().values().stream()
        .mapToDouble(SbmCharge.Result::total).sum();
    assertEquals(sbmFromClasses, r.sbm(), 1e-9);
    assertEquals(r.sbm() + r.drc() + r.rrao(), r.total(), 1e-9);

    assertTrue(r.perRiskClass().containsKey(RiskClass.EQUITY));
    assertTrue(r.perRiskClass().containsKey(RiskClass.GIRR));
    assertTrue(r.perRiskClass().containsKey(RiskClass.FX));
    assertEquals(r.total(), r.corep().total(), 1e-12);
    assertEquals(r.sbm(), r.corep().sbmTotal(), 1e-12);
    assertEquals("Basel MAR21", r.corep().parameterSet());

    // each class's corep row delta+vega+curvature equals its class charge
    r.corep().perRiskClass().forEach((rc, row) ->
        assertEquals(row.delta() + row.vega() + row.curvature(), row.classCharge(), 1e-9, rc + " row"));
  }

  @Test
  void dualRunsOneBookThroughTwoParameterSets() {
    Sensitivities book = EQ_BOOK;
    FrtbSa a = FrtbSa.of("USD").sbm(book, EQ_CURV);
    FrtbSa b = FrtbSa.of("USD").sbm(book, EQ_CURV);
    FrtbSa.Dual dual = FrtbSa.dual(a, b);
    assertEquals(dual.relieved().total(), dual.unrelieved().total(), 1e-12);
  }

  @Test
  void emptyBookIsZero() {
    FrtbSa.Result r = FrtbSa.of("EUR").compute();
    assertEquals(0.0, r.total(), 0.0);
    assertTrue(r.perRiskClass().isEmpty());
  }
}
