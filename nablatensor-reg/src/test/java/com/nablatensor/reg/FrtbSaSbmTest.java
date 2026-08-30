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

import com.nablatensor.reg.frtb.EquitySbmParameters;
import com.nablatensor.reg.frtb.FrtbSaSbm;
import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * FRTB SA-SBM equity, reconciled to a hand-worked example: an independent nested
 * aggregation written fresh in this test must match the calculator for every
 * correlation scenario, and the reported total must be the largest of the three.
 */
class FrtbSaSbmTest {

  private static RiskFactor d(String b, String n) {
    return RiskFactor.equityDelta(b, n);
  }

  private static RiskFactor v(String b, String n, double t) {
    return RiskFactor.equityVega(b, n, t);
  }

  private static final Sensitivities BOOK = Sensitivities.builder()
      .add(d("5", "ACME"), 1000.0)
      .add(d("5", "GLOBEX"), -400.0)
      .add(d("6", "INITECH"), 800.0)
      .add(v("5", "ACME", 1.0), 50.0)
      .add(v("5", "ACME", 2.0), -20.0)
      .add(v("6", "INITECH", 1.0), 15.0)
      .build();

  private static final List<FrtbSaSbm.CurvatureInput> CURV = List.of(
      new FrtbSaSbm.CurvatureInput(d("5", "ACME"), 100.0, 10.0, 200.0, 260.0, 170.0),
      new FrtbSaSbm.CurvatureInput(d("6", "INITECH"), 80.0, 6.0, 150.0, 140.0, 180.0));

  @Test
  void reconcilesToAnIndependentNestedAggregationPerScenario() {
    for (CorrelationScenario sc : CorrelationScenario.values()) {
      double refDelta = refDelta(sc);
      double refVega = refVega(sc);
      double refCurv = refCurv(sc);

      FrtbSaSbm.Result r = FrtbSaSbm.equity(BOOK, CURV);
      double got = r.perScenario().get(sc);
      assertEquals(refDelta + refVega + refCurv, got, 1e-8, "SA-SBM total under " + sc);
    }
  }

  @Test
  void totalIsTheLargestScenario() {
    FrtbSaSbm.Result r = FrtbSaSbm.equity(BOOK, CURV);
    double max = r.perScenario().values().stream().mapToDouble(Double::doubleValue).max().orElseThrow();
    assertEquals(max, r.total(), 1e-12, "reported total == max over correlation scenarios");
    assertEquals(r.perScenario().get(r.bindingScenario()), r.total(), 1e-12, "binding scenario is the max");
    assertTrue(r.total() > 0);
  }

  // ---- independent reference aggregation (delta / vega / curvature) -----

  private static double refDelta(CorrelationScenario sc) {
    RiskFactor[] f = {d("5", "ACME"), d("5", "GLOBEX"), d("6", "INITECH")};
    double[] s = {1000, -400, 800};
    double[] ws = new double[3];
    for (int i = 0; i < 3; i++) {
      ws[i] = EquitySbmParameters.deltaRiskWeight(f[i]) * s[i];
    }
    return nested(f, ws, sc, false, false);
  }

  private static double refVega(CorrelationScenario sc) {
    RiskFactor[] f = {v("5", "ACME", 1.0), v("5", "ACME", 2.0), v("6", "INITECH", 1.0)};
    double[] s = {50, -20, 15};
    double[] ws = new double[3];
    for (int i = 0; i < 3; i++) {
      ws[i] = EquitySbmParameters.vegaRiskWeight(f[i]) * s[i];
    }
    return nested(f, ws, sc, true, false);
  }

  private static double refCurv(CorrelationScenario sc) {
    RiskFactor a = d("5", "ACME");
    RiskFactor b = d("6", "INITECH");
    double rwA = EquitySbmParameters.curvatureRiskWeight(5);
    double rwB = EquitySbmParameters.curvatureRiskWeight(6);
    double cvrA = -Math.min(260 - 200 - rwA * 100 * 10, 170 - 200 + rwA * 100 * 10);
    double cvrB = -Math.min(140 - 150 - rwB * 80 * 6, 180 - 150 + rwB * 80 * 6);
    return nested(new RiskFactor[] {a.asCurvature(), b.asCurvature()}, new double[] {cvrA, cvrB}, sc, false, true);
  }

  /** Fresh nested aggregation, not using NestedAggregation. */
  private static double nested(RiskFactor[] f, double[] ws, CorrelationScenario sc, boolean vega, boolean curv) {
    java.util.Map<String, java.util.List<Integer>> byBucket = new java.util.LinkedHashMap<>();
    for (int i = 0; i < f.length; i++) {
      byBucket.computeIfAbsent(f[i].bucket(), x -> new java.util.ArrayList<>()).add(i);
    }
    java.util.Map<String, Double> kb = new java.util.LinkedHashMap<>();
    java.util.Map<String, Double> sb = new java.util.LinkedHashMap<>();
    for (var e : byBucket.entrySet()) {
      var idx = e.getValue();
      double sum = 0;
      double sw = 0;
      for (int a : idx) {
        sw += ws[a];
        sum += curv ? Math.pow(Math.max(ws[a], 0), 2) : ws[a] * ws[a];
        for (int b : idx) {
          if (a == b) {
            continue;
          }
          double rho = sc.apply(vega ? EquitySbmParameters.vegaRho(f[a], f[b]) : EquitySbmParameters.deltaRho(f[a], f[b]));
          double psi = curv && ws[a] < 0 && ws[b] < 0 ? 0 : 1;
          sum += (curv ? rho * rho : rho) * ws[a] * ws[b] * psi;
        }
      }
      double k = Math.sqrt(Math.max(0, sum));
      kb.put(e.getKey(), k);
      sb.put(e.getKey(), Math.max(-k, Math.min(sw, k)));
    }
    var bs = new java.util.ArrayList<>(kb.keySet());
    double t = 0;
    for (int i = 0; i < bs.size(); i++) {
      t += kb.get(bs.get(i)) * kb.get(bs.get(i));
      for (int j = 0; j < bs.size(); j++) {
        if (i == j) {
          continue;
        }
        double g = sc.apply(EquitySbmParameters.gamma(bs.get(i), bs.get(j)));
        double si = sb.get(bs.get(i));
        double sj = sb.get(bs.get(j));
        double psi = curv && si < 0 && sj < 0 ? 0 : 1;
        t += (curv ? g * g : g) * si * sj * psi;
      }
    }
    return Math.sqrt(Math.max(0, t));
  }
}
