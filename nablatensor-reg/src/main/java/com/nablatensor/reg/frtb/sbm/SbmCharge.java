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
package com.nablatensor.reg.frtb.sbm;

import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.NestedAggregation;
import com.nablatensor.risk.RiskClassProfile;
import com.nablatensor.risk.RiskMeasure;
import com.nablatensor.risk.Sensitivities;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The FRTB SA Sensitivities-Based Method for <b>one risk class</b> (MAR21),
 * generic over a {@link RiskClassProfile}. Delta + vega + curvature, each
 * aggregated with {@link NestedAggregation} under the three
 * {@link CorrelationScenario}s; the class charge is the largest of the three.
 *
 * <p>This is the equity-only {@code FrtbSaSbm} generalised: same aggregation,
 * same curvature formula, same "max over three scenarios", but the parameter
 * tables come from the profile so GIRR / CSR / commodity / FX drop in unchanged.
 *
 * <p>Calculators, not sign-off.
 */
public final class SbmCharge {

  /** Per-class SBM result: the components at the binding scenario and the per-scenario totals. */
  public record Result(double delta, double vega, double curvature, double total,
                       CorrelationScenario bindingScenario,
                       Map<CorrelationScenario, Double> perScenario) {
  }

  private final RiskClassProfile profile;

  private SbmCharge(RiskClassProfile profile) {
    this.profile = profile;
  }

  public static SbmCharge of(RiskClassProfile profile) {
    return new SbmCharge(profile);
  }

  /**
   * @param book    all sensitivities of this risk class (delta and vega factors); other classes are ignored
   * @param curv    curvature repricings for this risk class ({@code null} / empty allowed)
   */
  public Result compute(Sensitivities book, List<CurvatureRepricing> curv) {
    Sensitivities cls = book.ofClass(profile.riskClass());
    Sensitivities deltas = cls.ofMeasure(RiskMeasure.DELTA);
    Sensitivities vegas = cls.ofMeasure(RiskMeasure.VEGA);

    Map<CorrelationScenario, Double> totals = new LinkedHashMap<>();
    double bestD = 0;
    double bestV = 0;
    double bestC = 0;
    double best = Double.NEGATIVE_INFINITY;
    CorrelationScenario binding = CorrelationScenario.MEDIUM;

    for (CorrelationScenario sc : CorrelationScenario.values()) {
      double d = aggregate(deltas, sc, RiskMeasure.DELTA);
      double v = aggregate(vegas, sc, RiskMeasure.VEGA);
      double c = curvature(curv, sc);
      double t = d + v + c;
      totals.put(sc, t);
      if (t > best) {
        best = t;
        bestD = d;
        bestV = v;
        bestC = c;
        binding = sc;
      }
    }
    return new Result(bestD, bestV, bestC, best, binding, totals);
  }

  private double aggregate(Sensitivities s, CorrelationScenario sc, RiskMeasure measure) {
    if (s.isEmpty()) {
      return 0.0;
    }
    NestedAggregation.RiskWeight rw = measure == RiskMeasure.VEGA
        ? profile::vegaRiskWeight
        : profile::deltaRiskWeight;
    NestedAggregation.WithinBucketCorrelation rho = measure == RiskMeasure.VEGA
        ? (k, l) -> sc.apply(profile.vegaRho(k, l))
        : (k, l) -> sc.apply(profile.deltaRho(k, l));
    NestedAggregation agg = NestedAggregation.delta(rw, rho,
        (b, c) -> sc.apply(profile.gamma(b, c)));
    return agg.aggregate(s).total();
  }

  private double curvature(List<CurvatureRepricing> inputs, CorrelationScenario sc) {
    if (inputs == null || inputs.isEmpty()) {
      return 0.0;
    }
    Sensitivities.Builder cvr = Sensitivities.builder();
    for (CurvatureRepricing in : inputs) {
      if (in.factor().riskClass() != profile.riskClass()) {
        continue;
      }
      double shock = profile.curvatureShock(in.factor(), in.riskFactorLevel());
      double up = in.pvUp() - in.pvBase() - shock * in.netDelta();
      double down = in.pvDown() - in.pvBase() + shock * in.netDelta();
      double cvrK = -Math.min(up, down);
      cvr.add(in.factor().asCurvature(), cvrK);
    }
    Sensitivities built = cvr.build();
    if (built.isEmpty()) {
      return 0.0;
    }
    NestedAggregation agg = NestedAggregation.curvature(
        (k, l) -> sc.apply(profile.deltaRho(k, l)),
        (b, c) -> sc.apply(profile.gamma(b, c)));
    return agg.aggregate(built).total();
  }
}
