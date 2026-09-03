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
package com.nablatensor.cva;

import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.NestedAggregation;
import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskMeasure;
import com.nablatensor.risk.Sensitivities;
import java.util.EnumMap;
import java.util.Map;

/**
 * The SA-CVA capital charge from a bucketed CVA sensitivity vector. Within each
 * risk type the delta and vega sensitivities are aggregated FRTB-style on the
 * shared {@link NestedAggregation}; the risk types are then combined as
 * {@code sqrt(sum of squares)} and scaled by {@code m_CVA}. The charge is
 * computed under all three correlation scenarios and the largest is kept
 * (MAR50.9, CRR3 Art. 383).
 */
public final class SaCva {

  private static final RiskClass[] RISK_TYPES = {
      RiskClass.GIRR, RiskClass.CSR_NON_SEC, RiskClass.FX
  };

  private final SaCvaParameters parameters;

  public SaCva(SaCvaParameters parameters) {
    this.parameters = parameters;
  }

  public SaCvaResult charge(Sensitivities cvaSensitivities) {
    Map<CorrelationScenario, Double> perScenario = new EnumMap<>(CorrelationScenario.class);
    Map<CorrelationScenario, Map<RiskClass, Double>> byTypeByScenario =
        new EnumMap<>(CorrelationScenario.class);

    for (CorrelationScenario scenario : CorrelationScenario.values()) {
      Map<RiskClass, Double> byType = new EnumMap<>(RiskClass.class);
      double sumOfSquares = 0.0;
      for (RiskClass riskType : RISK_TYPES) {
        double delta = aggregate(cvaSensitivities, riskType, RiskMeasure.DELTA, scenario, false);
        double vega = aggregate(cvaSensitivities, riskType, RiskMeasure.VEGA, scenario, true);
        double kRiskType = Math.hypot(delta, vega);
        byType.put(riskType, kRiskType);
        sumOfSquares += kRiskType * kRiskType;
      }
      byTypeByScenario.put(scenario, byType);
      perScenario.put(scenario, parameters.mCva() * Math.sqrt(sumOfSquares));
    }

    CorrelationScenario selected = perScenario.entrySet().stream()
        .max(Map.Entry.comparingByValue()).orElseThrow().getKey();
    return new SaCvaResult(perScenario, byTypeByScenario.get(selected), selected, perScenario.get(selected));
  }

  private double aggregate(Sensitivities all, RiskClass riskType, RiskMeasure measure,
                           CorrelationScenario scenario, boolean vega) {
    Sensitivities slice = all.ofClass(riskType).ofMeasure(measure);
    if (slice.isEmpty()) {
      return 0.0;
    }
    double withinRho = scenario.apply(parameters.withinBucketCorrelation(riskType));
    double acrossGamma = scenario.apply(parameters.acrossBucketCorrelation(riskType));
    NestedAggregation aggregation = NestedAggregation.delta(
        factor -> vega ? parameters.vegaRiskWeight(factor) : parameters.deltaRiskWeight(factor),
        (left, right) -> left.equals(right) ? 1.0 : withinRho,
        (bucketB, bucketC) -> bucketB.equals(bucketC) ? 1.0 : acrossGamma);
    return aggregation.aggregate(slice).total();
  }
}
