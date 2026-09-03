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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The BA-CVA capital charge (Basel MAR50.1-50.8; CRR3 Art. 384).
 *
 * <p><b>Reduced</b>:
 * <pre>{@code
 * SCVA_c    = ( RW_c * M_c * EAD_c * DF_c ) / alpha
 * K_reduced = sqrt( ( rho * sum_c SCVA_c )^2 + (1 - rho^2) * sum_c SCVA_c^2 )
 * }</pre>
 *
 * <p><b>Full</b> adds single-name and index CDS hedge recognition:
 * <pre>{@code
 * K_hedged = sqrt( ( rho * sum_c (SCVA_c - SNH_c) - IH )^2
 *                  + (1 - rho^2) * sum_c (SCVA_c - SNH_c)^2
 *                  + sum_c HMA_c )
 * K_full   = beta * K_reduced + (1 - beta) * K_hedged
 * }</pre>
 *
 * with {@code SNH_c} the single-name hedges of counterparty {@code c} weighted by
 * {@code r_hc}, {@code IH} the index hedges, and {@code HMA_c} the
 * hedge-misalignment residual {@code sum_h (1 - r_hc^2) * (RW_h M_h B_h DF_h)^2}.
 */
public final class BaCva {

  private final BaCvaParameters parameters;

  public BaCva(BaCvaParameters parameters) {
    this.parameters = parameters;
  }

  /** One counterparty's inputs to the charge. */
  public record Exposure(CreditName counterparty, double effectiveMaturityYears,
                         double exposureAtDefault) {}

  public BaCvaResult charge(List<Exposure> exposures, List<CvaHedge> hedges) {
    double rho = parameters.rho();
    Map<String, Double> scvaByCounterparty = new LinkedHashMap<>();
    Map<String, Double> singleNameHedgeByCounterparty = new LinkedHashMap<>();
    Map<String, Double> hedgeMisalignmentByCounterparty = new LinkedHashMap<>();

    for (Exposure exposure : exposures) {
      double riskWeight = parameters.riskWeight(
          exposure.counterparty().rating(), exposure.counterparty().sector());
      double discount = parameters.supervisoryDiscount(exposure.effectiveMaturityYears());
      double scva = riskWeight * exposure.effectiveMaturityYears()
          * exposure.exposureAtDefault() * discount / parameters.alpha();
      scvaByCounterparty.merge(exposure.counterparty().id(), scva, Double::sum);
    }

    double indexHedge = 0.0;
    for (CvaHedge hedge : hedges) {
      double magnitude = hedge.discountedWeightedNotional();
      if (hedge.kind() == CvaHedge.Kind.INDEX_CDS) {
        indexHedge += magnitude;
      } else {
        singleNameHedgeByCounterparty.merge(hedge.referenceId(),
            hedge.correlation() * magnitude, Double::sum);
        hedgeMisalignmentByCounterparty.merge(hedge.referenceId(),
            (1.0 - hedge.correlation() * hedge.correlation()) * magnitude * magnitude, Double::sum);
      }
    }

    double reduced = hedgedCharge(scvaByCounterparty, Map.of(), Map.of(), 0.0, rho);

    double full = parameters.beta() * reduced + (1.0 - parameters.beta())
        * hedgedCharge(scvaByCounterparty, singleNameHedgeByCounterparty,
            hedgeMisalignmentByCounterparty, indexHedge, rho);

    return new BaCvaResult(reduced, full, scvaByCounterparty, reduced - full);
  }

  private static double hedgedCharge(Map<String, Double> scva, Map<String, Double> snh,
                                     Map<String, Double> hma, double indexHedge, double rho) {
    double weightedSum = 0.0;
    double sumOfSquares = 0.0;
    double misalignment = 0.0;
    for (Map.Entry<String, Double> entry : scva.entrySet()) {
      double net = entry.getValue() - snh.getOrDefault(entry.getKey(), 0.0);
      weightedSum += net;
      sumOfSquares += net * net;
      misalignment += hma.getOrDefault(entry.getKey(), 0.0);
    }
    double systematic = rho * weightedSum - indexHedge;
    return Math.sqrt(systematic * systematic + (1.0 - rho * rho) * sumOfSquares + misalignment);
  }
}
