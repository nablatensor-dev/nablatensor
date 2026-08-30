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
package com.nablatensor.reg.frtb;

import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.NestedAggregation;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.RiskMeasure;
import com.nablatensor.risk.Sensitivities;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * FRTB SA Sensitivities-Based Method for the <b>equity</b> risk class
 * (MAR21). Delta + vega + curvature, each aggregated with
 * {@link NestedAggregation} under the three correlation scenarios; the charge is
 * the largest of the three.
 *
 * <p>The delta and vega vectors are exactly what one adjoint sweep of the book
 * produces, mapped onto {@link RiskFactor}s. Curvature needs the two shocked
 * repricings per name — {@link CurvatureInput} carries them; wire them from a
 * {@code MonteCarlo} + {@code setInput} as in the docs.
 *
 * <p>Ships {@code calculators}, not sign-off: it computes the number MAR21 asks
 * for; validation and submission are the user's.
 */
public final class FrtbSaSbm {

  /** Per-equity-name curvature inputs: base PV and the two shocked PVs, plus the book delta and spot. */
  public record CurvatureInput(RiskFactor factor, double spot, double netDelta,
                               double pvBase, double pvUp, double pvDown) {}

  /** The equity SA-SBM result: the components and the scenario that binds. */
  public record Result(double delta, double vega, double curvature, double total,
                       CorrelationScenario bindingScenario,
                       Map<CorrelationScenario, Double> perScenario) {}

  private FrtbSaSbm() {
  }

  public static Result equity(Sensitivities bookSensitivities, List<CurvatureInput> curvature) {
    Sensitivities eq = bookSensitivities.ofClass(com.nablatensor.risk.RiskClass.EQUITY);
    Sensitivities deltas = eq.ofMeasure(RiskMeasure.DELTA);
    Sensitivities vegas = eq.ofMeasure(RiskMeasure.VEGA);

    Map<CorrelationScenario, Double> totals = new LinkedHashMap<>();
    double bestDelta = 0;
    double bestVega = 0;
    double bestCurv = 0;
    double best = Double.NEGATIVE_INFINITY;
    CorrelationScenario binding = CorrelationScenario.MEDIUM;

    for (CorrelationScenario sc : CorrelationScenario.values()) {
      double d = aggregate(deltas, sc, false, RiskMeasure.DELTA);
      double v = aggregate(vegas, sc, false, RiskMeasure.VEGA);
      double c = curvature(curvature, sc);
      double t = d + v + c;
      totals.put(sc, t);
      if (t > best) {
        best = t;
        bestDelta = d;
        bestVega = v;
        bestCurv = c;
        binding = sc;
      }
    }
    return new Result(bestDelta, bestVega, bestCurv, best, binding, totals);
  }

  private static double aggregate(Sensitivities s, CorrelationScenario sc, boolean curv, RiskMeasure measure) {
    if (s.isEmpty()) {
      return 0.0;
    }
    NestedAggregation.WithinBucketCorrelation rho = measure == RiskMeasure.VEGA
        ? (k, l) -> sc.apply(EquitySbmParameters.vegaRho(k, l))
        : (k, l) -> sc.apply(EquitySbmParameters.deltaRho(k, l));
    NestedAggregation.RiskWeight rw = measure == RiskMeasure.VEGA
        ? EquitySbmParameters::vegaRiskWeight
        : EquitySbmParameters::deltaRiskWeight;
    NestedAggregation agg = NestedAggregation.delta(rw, rho,
        (b, c) -> sc.apply(EquitySbmParameters.gamma(b, c)));
    return agg.aggregate(s).total();
  }

  private static double curvature(List<CurvatureInput> inputs, CorrelationScenario sc) {
    if (inputs == null || inputs.isEmpty()) {
      return 0.0;
    }
    Sensitivities.Builder cvr = Sensitivities.builder();
    for (CurvatureInput in : inputs) {
      double rw = EquitySbmParameters.curvatureRiskWeight(EquitySbmParameters.bucket(in.factor()));
      double up = in.pvUp() - in.pvBase() - rw * in.spot() * in.netDelta();
      double down = in.pvDown() - in.pvBase() + rw * in.spot() * in.netDelta();
      double cvrK = -Math.min(up, down);
      cvr.add(in.factor().asCurvature(), cvrK);
    }
    NestedAggregation agg = NestedAggregation.curvature(
        (k, l) -> sc.apply(EquitySbmParameters.deltaRho(k, l)),
        (b, c) -> sc.apply(EquitySbmParameters.gamma(b, c)));
    return agg.aggregate(cvr.build()).total();
  }

  /** Convenience: the risk factors present, for wiring the curvature repricings. */
  public static List<RiskFactor> deltaFactors(Sensitivities book) {
    List<RiskFactor> out = new ArrayList<>();
    book.ofClass(com.nablatensor.risk.RiskClass.EQUITY).ofMeasure(RiskMeasure.DELTA)
        .asMap().keySet().forEach(out::add);
    return out;
  }
}
