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

import com.nablatensor.reg.frtb.sbm.CurvatureRepricing;
import com.nablatensor.reg.frtb.sbm.SbmCharge;
import com.nablatensor.reg.frtb.sbm.commodity.CommoditySbmParameters;
import com.nablatensor.reg.frtb.sbm.csr.CsrSbmParameters;
import com.nablatensor.reg.frtb.sbm.fx.FxSbmParameters;
import com.nablatensor.reg.frtb.sbm.girr.GirrSbmParameters;
import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.RiskClassProfile;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.RiskMeasure;
import com.nablatensor.risk.Sensitivities;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * The generic {@link SbmCharge} engine reconciled, for GIRR / CSR non-sec /
 * commodity / FX, to an <b>independent</b> nested aggregation written fresh in
 * this test (not calling {@code NestedAggregation}) — for every correlation
 * scenario, and the reported total is the largest of the three.
 */
class SbmChargeTest {

  // ---------------------------------------------------------------- GIRR

  private static final GirrSbmParameters GIRR = GirrSbmParameters.baselDefault();

  private static final Sensitivities GIRR_BOOK = Sensitivities.builder()
      .add(RiskFactor.girrDelta("EUR", "OIS", 2), 12_000.0)
      .add(RiskFactor.girrDelta("EUR", "OIS", 5), -4_500.0)
      .add(RiskFactor.girrDelta("EUR", "OIS", 10), 8_000.0)
      .add(RiskFactor.girrDelta("EUR", "3M", 5), 1_200.0)
      .add(RiskFactor.girrInflation("EUR"), 900.0)
      .add(RiskFactor.girrDelta("USD", "OIS", 2), -6_000.0)
      .add(RiskFactor.girrDelta("USD", "OIS", 10), 3_300.0)
      .add(RiskFactor.girrXccyBasis("USD"), 500.0)
      .add(RiskFactor.girrVega("EUR", 1, 5), 250.0)
      .add(RiskFactor.girrVega("EUR", 5, 5), -80.0)
      .build();

  private static final List<CurvatureRepricing> GIRR_CURV = List.of(
      new CurvatureRepricing(RiskFactor.girrDelta("EUR", "OIS", 0), 0.03, 15_500.0,
          1_000_000.0, 999_400.0, 1_000_650.0),
      new CurvatureRepricing(RiskFactor.girrDelta("USD", "OIS", 0), 0.03, -2_700.0,
          1_000_000.0, 1_000_120.0, 999_970.0));

  @Test
  void girrReconcilesPerScenario() {
    reconcile(GIRR, GIRR_BOOK, GIRR_CURV);
  }

  // ---------------------------------------------------------------- CSR non-sec

  private static final CsrSbmParameters CSR = CsrSbmParameters.nonSec();

  private static final Sensitivities CSR_BOOK = Sensitivities.builder()
      .add(RiskFactor.csrDelta("3", "ACME", RiskFactor.CsrCurve.BOND, 1), 5_000.0)
      .add(RiskFactor.csrDelta("3", "ACME", RiskFactor.CsrCurve.CDS, 1), -3_000.0)
      .add(RiskFactor.csrDelta("3", "ACME", RiskFactor.CsrCurve.BOND, 5), 2_000.0)
      .add(RiskFactor.csrDelta("3", "GLOBEX", RiskFactor.CsrCurve.BOND, 5), 1_500.0)
      .add(RiskFactor.csrDelta("11", "HYCO", RiskFactor.CsrCurve.CDS, 3), -4_000.0)
      .add(RiskFactor.csrVega("3", "ACME", 1), 120.0)
      .add(RiskFactor.csrVega("3", "ACME", 5), -40.0)
      .build();

  @Test
  void csrNonSecReconcilesPerScenario() {
    reconcile(CSR, CSR_BOOK, List.of());
  }

  // ---------------------------------------------------------------- commodity

  private static final CommoditySbmParameters CTY = CommoditySbmParameters.INSTANCE;

  private static final Sensitivities CTY_BOOK = Sensitivities.builder()
      .add(RiskFactor.commodityDelta("2", "WTI", 1, "CUSHING"), 7_000.0)
      .add(RiskFactor.commodityDelta("2", "WTI", 3, "CUSHING"), -2_500.0)
      .add(RiskFactor.commodityDelta("2", "BRENT", 1, "SULLOM"), 3_000.0)
      .add(RiskFactor.commodityDelta("5", "COPPER", 1, "LME"), 4_000.0)
      .build();

  private static final List<CurvatureRepricing> CTY_CURV = List.of(
      new CurvatureRepricing(RiskFactor.commodityDelta("2", "WTI", 1, "CUSHING"), 80.0, 55.0,
          500_000.0, 505_000.0, 496_000.0));

  @Test
  void commodityReconcilesPerScenario() {
    reconcile(CTY, CTY_BOOK, CTY_CURV);
  }

  // ---------------------------------------------------------------- FX

  private static final FxSbmParameters FX = FxSbmParameters.INSTANCE;

  private static final Sensitivities FX_BOOK = Sensitivities.builder()
      .add(RiskFactor.fxDelta("EURUSD"), 1_000_000.0)
      .add(RiskFactor.fxDelta("EURJPY"), -400_000.0)
      .add(RiskFactor.fxDelta("EURZAR"), 250_000.0)
      .build();

  @Test
  void fxReconcilesPerScenario() {
    reconcile(FX, FX_BOOK, List.of());
  }

  // ---------------------------------------------------------------- shared

  private static void reconcile(RiskClassProfile p, Sensitivities book, List<CurvatureRepricing> curv) {
    SbmCharge.Result r = SbmCharge.of(p).compute(book, curv);
    Map<CorrelationScenario, Double> ref = new LinkedHashMap<>();
    for (CorrelationScenario sc : CorrelationScenario.values()) {
      double d = refDeltaVega(p, book.ofClass(p.riskClass()).ofMeasure(RiskMeasure.DELTA), sc, false);
      double v = refDeltaVega(p, book.ofClass(p.riskClass()).ofMeasure(RiskMeasure.VEGA), sc, true);
      double c = refCurvature(p, curv, sc);
      double total = d + v + c;
      ref.put(sc, total);
      assertEquals(total, r.perScenario().get(sc), 1e-6, p.riskClass() + " SBM total under " + sc);
    }
    double max = ref.values().stream().mapToDouble(Double::doubleValue).max().orElseThrow();
    assertEquals(max, r.total(), 1e-9, p.riskClass() + " total == max over scenarios");
    assertEquals(ref.get(r.bindingScenario()), r.total(), 1e-9, "binding scenario is the max");
    assertTrue(r.total() >= 0.0);
  }

  /** Fresh nested aggregation for delta or vega — does not use NestedAggregation. */
  private static double refDeltaVega(RiskClassProfile p, Sensitivities s, CorrelationScenario sc, boolean vega) {
    if (s.isEmpty()) {
      return 0.0;
    }
    Map<String, List<RiskFactor>> byBucket = new LinkedHashMap<>();
    Map<RiskFactor, Double> ws = new LinkedHashMap<>();
    for (var e : s.asMap().entrySet()) {
      RiskFactor f = e.getKey();
      double rw = vega ? p.vegaRiskWeight(f) : p.deltaRiskWeight(f);
      ws.put(f, rw * e.getValue());
      byBucket.computeIfAbsent(f.bucket(), b -> new ArrayList<>()).add(f);
    }
    Map<String, Double> kb = new LinkedHashMap<>();
    Map<String, Double> sb = new LinkedHashMap<>();
    for (var e : byBucket.entrySet()) {
      List<RiskFactor> fs = e.getValue();
      double sum = 0;
      double sw = 0;
      for (int i = 0; i < fs.size(); i++) {
        double wi = ws.get(fs.get(i));
        sw += wi;
        sum += wi * wi;
        for (int j = 0; j < fs.size(); j++) {
          if (i == j) {
            continue;
          }
          double wj = ws.get(fs.get(j));
          double rho = sc.apply(vega ? p.vegaRho(fs.get(i), fs.get(j)) : p.deltaRho(fs.get(i), fs.get(j)));
          sum += rho * wi * wj;
        }
      }
      double k = Math.sqrt(Math.max(0, sum));
      kb.put(e.getKey(), k);
      sb.put(e.getKey(), Math.max(-k, Math.min(sw, k)));
    }
    return acrossBuckets(p, kb, sb, sc, false);
  }

  /** Fresh nested aggregation for curvature — does not use NestedAggregation. */
  private static double refCurvature(RiskClassProfile p, List<CurvatureRepricing> inputs, CorrelationScenario sc) {
    List<CurvatureRepricing> mine = new ArrayList<>();
    for (CurvatureRepricing in : inputs) {
      if (in.factor().riskClass() == p.riskClass()) {
        mine.add(in);
      }
    }
    if (mine.isEmpty()) {
      return 0.0;
    }
    Map<String, List<RiskFactor>> byBucket = new LinkedHashMap<>();
    Map<RiskFactor, Double> cvr = new LinkedHashMap<>();
    for (CurvatureRepricing in : mine) {
      double shock = p.curvatureShock(in.factor(), in.riskFactorLevel());
      double up = in.pvUp() - in.pvBase() - shock * in.netDelta();
      double down = in.pvDown() - in.pvBase() + shock * in.netDelta();
      RiskFactor cf = in.factor().asCurvature();
      cvr.put(cf, -Math.min(up, down));
      byBucket.computeIfAbsent(cf.bucket(), b -> new ArrayList<>()).add(cf);
    }
    Map<String, Double> kb = new LinkedHashMap<>();
    Map<String, Double> sb = new LinkedHashMap<>();
    for (var e : byBucket.entrySet()) {
      List<RiskFactor> fs = e.getValue();
      double sum = 0;
      double sw = 0;
      for (int i = 0; i < fs.size(); i++) {
        double wi = cvr.get(fs.get(i));
        sw += wi;
        sum += Math.pow(Math.max(wi, 0), 2);
        for (int j = 0; j < fs.size(); j++) {
          if (i == j) {
            continue;
          }
          double wj = cvr.get(fs.get(j));
          double rho = sc.apply(p.deltaRho(fs.get(i), fs.get(j)));
          double psi = wi < 0 && wj < 0 ? 0 : 1;
          sum += rho * rho * wi * wj * psi;
        }
      }
      double k = Math.sqrt(Math.max(0, sum));
      kb.put(e.getKey(), k);
      sb.put(e.getKey(), Math.max(-k, Math.min(sw, k)));
    }
    return acrossBuckets(p, kb, sb, sc, true);
  }

  private static double acrossBuckets(RiskClassProfile p, Map<String, Double> kb, Map<String, Double> sb,
                                      CorrelationScenario sc, boolean curv) {
    List<String> bs = new ArrayList<>(kb.keySet());
    double t = 0;
    for (int i = 0; i < bs.size(); i++) {
      t += kb.get(bs.get(i)) * kb.get(bs.get(i));
      for (int j = 0; j < bs.size(); j++) {
        if (i == j) {
          continue;
        }
        double g = sc.apply(p.gamma(bs.get(i), bs.get(j)));
        double si = sb.get(bs.get(i));
        double sj = sb.get(bs.get(j));
        double psi = curv && si < 0 && sj < 0 ? 0 : 1;
        t += (curv ? g * g : g) * si * sj * psi;
      }
    }
    return Math.sqrt(Math.max(0, t));
  }
}
