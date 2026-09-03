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

import com.nablatensor.risk.Sensitivities;
import java.util.List;
import java.util.Map;

/**
 * The full CVA picture for a portfolio of netting sets: the exposure-simulation
 * result per netting set, the aggregated CVA risk vector from the adjoint
 * sweeps, and the BA-CVA and SA-CVA capital charges.
 *
 * @param cvaValue        total unilateral CVA across the portfolio (reporting currency)
 * @param perNettingSet   the {@link CvaResult} for each netting set, in input order
 * @param aggregateSensitivities the netted SA-CVA sensitivity vector
 * @param baCva           the BA-CVA charge (reduced and full)
 * @param saCva           the SA-CVA charge and its build-up
 * @param sweepSeconds    total adjoint-sweep wall time across the netting sets
 * @param bumpRevaluations exposure re-simulations a prescribed-bump SA-CVA vector would have cost
 */
public record CvaCapital(double cvaValue,
                         List<CvaResult> perNettingSet,
                         Sensitivities aggregateSensitivities,
                         BaCvaResult baCva,
                         SaCvaResult saCva,
                         double sweepSeconds,
                         int bumpRevaluations) {

  public CvaCapital {
    perNettingSet = List.copyOf(perNettingSet);
  }

  /** {@code SCVA_c} contributions behind the BA-CVA reduced charge. */
  public Map<String, Double> scvaByCounterparty() {
    return baCva.scvaByCounterparty();
  }
}
