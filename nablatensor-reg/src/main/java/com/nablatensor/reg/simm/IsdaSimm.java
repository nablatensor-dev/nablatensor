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
package com.nablatensor.reg.simm;

import com.nablatensor.risk.NestedAggregation;
import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskMeasure;
import com.nablatensor.risk.Sensitivities;

/**
 * ISDA SIMM initial-margin aggregation for the <b>equity</b> risk class: delta
 * margin + vega margin, each with the concentration risk factor
 * {@code CR_k = max(1, sqrt(|s_k| / T_b))} feeding both the weighted sensitivity
 * and the {@code f_kl} / {@code g_bc} correlation corrections
 * ({@link NestedAggregation#withConcentration}).
 *
 * <p>Uses {@link SimmEquityParameters} — see its warning: the numbers there are
 * illustrative, not the ISDA calibration.
 */
public final class IsdaSimm {

  public record Result(double deltaMargin, double vegaMargin, double total) {}

  private IsdaSimm() {
  }

  public static Result equity(Sensitivities book) {
    Sensitivities eq = book.ofClass(RiskClass.EQUITY);

    NestedAggregation delta = NestedAggregation.delta(
            SimmEquityParameters::deltaRiskWeight,
            SimmEquityParameters::withinBucketRho,
            SimmEquityParameters::acrossBucketGamma)
        .withConcentration((f, s) -> Math.max(1.0,
            Math.sqrt(Math.abs(s) / SimmEquityParameters.deltaConcentrationThreshold(f))));
    double dm = delta.aggregate(eq.ofMeasure(RiskMeasure.DELTA)).total();

    NestedAggregation vega = NestedAggregation.delta(
            f -> SimmEquityParameters.vegaRiskWeight() * SimmEquityParameters.historicalVolatilityRatio(),
            SimmEquityParameters::withinBucketRho,
            SimmEquityParameters::acrossBucketGamma)
        .withConcentration((f, s) -> Math.max(1.0,
            Math.sqrt(Math.abs(s) / SimmEquityParameters.vegaConcentrationThreshold())));
    double vm = vega.aggregate(eq.ofMeasure(RiskMeasure.VEGA)).total();

    return new Result(dm, vm, dm + vm);
  }
}
