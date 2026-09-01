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

import com.nablatensor.reg.frtb.sa.FrtbParameterSet;
import com.nablatensor.reg.frtb.sa.FrtbSa;
import com.nablatensor.reg.frtb.sa.ReportingCurrency;
import com.nablatensor.reg.frtb.sbm.girr.GirrSbmParameters;
import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskClassProfile;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * LESSON 20 — the CRR3 "relieved vs unrelieved" COREP dual. Reporting the FRTB
 * number on two bases (with and without a jurisdiction's transitional relief) is
 * <b>mechanically two runs of the same assembler with two parameter sets</b> —
 * one book, no recompile:
 *
 * <pre>
 *   book ──►  FrtbSa(unrelieved params)  ──►  own funds (headline)
 *        └──►  FrtbSa(relieved params)    ──►  own funds (with relief)
 * </pre>
 *
 * Here "relief" is a toy 20% scaling of the GIRR delta risk weights — a stand-in
 * for the EU targeted multiplier (the real relief parameters are a later phase).
 */
class Lesson20_RelievedUnrelievedDualTest {

  /** A profile that scales another profile's delta risk weight — a toy relief multiplier. */
  private static RiskClassProfile scaledDelta(RiskClassProfile base, double factor) {
    return new RiskClassProfile() {
      @Override public RiskClass riskClass() {
        return base.riskClass();
      }

      @Override public double deltaRiskWeight(RiskFactor k) {
        return base.deltaRiskWeight(k) * factor;
      }

      @Override public double vegaRiskWeight(RiskFactor k) {
        return base.vegaRiskWeight(k);
      }

      @Override public double curvatureShock(RiskFactor k, double level) {
        return base.curvatureShock(k, level);
      }

      @Override public double deltaRho(RiskFactor k, RiskFactor l) {
        return base.deltaRho(k, l);
      }

      @Override public double vegaRho(RiskFactor k, RiskFactor l) {
        return base.vegaRho(k, l);
      }

      @Override public double gamma(String b, String c) {
        return base.gamma(b, c);
      }
    };
  }

  private static FrtbParameterSet reliefSet() {
    FrtbParameterSet basel = FrtbParameterSet.baselMar21();
    Map<RiskClass, RiskClassProfile> m = new EnumMap<>(RiskClass.class);
    for (RiskClass rc : RiskClass.values()) {
      m.put(rc, basel.profile(rc));
    }
    m.put(RiskClass.GIRR, scaledDelta(GirrSbmParameters.baselDefault(), 0.80));
    return FrtbParameterSet.of("EU relief (toy)", m);
  }

  @Test
  void reliefLowersTheGirrDrivenChargeButNotByTouchingTheBook() {
    Sensitivities book = Sensitivities.builder()
        .add(RiskFactor.girrDelta("EUR", "OIS", 2), 20_000.0)
        .add(RiskFactor.girrDelta("EUR", "OIS", 10), 12_000.0)
        .build();

    FrtbSa unrelieved = FrtbSa.of(FrtbParameterSet.baselMar21(), ReportingCurrency.of("EUR")).sbm(book, List.of());
    FrtbSa relieved = FrtbSa.of(reliefSet(), ReportingCurrency.of("EUR")).sbm(book, List.of());

    FrtbSa.Dual dual = FrtbSa.dual(relieved, unrelieved);

    assertTrue(dual.relieved().total() < dual.unrelieved().total());
    // GIRR delta scales linearly with the risk weight, so the relieved charge is 80% of the headline
    assertEquals(0.80, dual.relieved().total() / dual.unrelieved().total(), 1e-9);
    assertEquals("EU relief (toy)", dual.relieved().corep().parameterSet());
    assertEquals("Basel MAR21", dual.unrelieved().corep().parameterSet());
  }
}
