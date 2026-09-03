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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class BaCvaTest {

  private static final BaCvaParameters PARAMETERS = BaCvaParameters.standard();

  private static CreditName name(String id, CreditName.Rating rating, CreditName.Sector sector) {
    return new CreditName(id, HazardCurve.fromFlatSpread(150.0, 0.4, 10.0), 0.4, rating, sector);
  }

  @Test
  void reducedMatchesTheClosedForm() {
    BaCva baCva = new BaCva(PARAMETERS);
    List<BaCva.Exposure> exposures = List.of(
        new BaCva.Exposure(name("A", CreditName.Rating.A, CreditName.Sector.FINANCIAL), 4.0, 20_000_000.0),
        new BaCva.Exposure(name("B", CreditName.Rating.BBB, CreditName.Sector.CORPORATE), 3.0, 15_000_000.0));

    double[] scva = new double[2];
    for (int i = 0; i < exposures.size(); i++) {
      BaCva.Exposure e = exposures.get(i);
      double rw = PARAMETERS.riskWeight(e.counterparty().rating(), e.counterparty().sector());
      double df = PARAMETERS.supervisoryDiscount(e.effectiveMaturityYears());
      scva[i] = rw * e.effectiveMaturityYears() * e.exposureAtDefault() * df / PARAMETERS.alpha();
    }
    double rho = PARAMETERS.rho();
    double systematic = rho * (scva[0] + scva[1]);
    double expected = Math.sqrt(systematic * systematic
        + (1.0 - rho * rho) * (scva[0] * scva[0] + scva[1] * scva[1]));

    BaCvaResult result = baCva.charge(exposures, List.of());
    assertEquals(expected, result.reduced(), 1.0e-6 * expected);
    assertEquals(scva[0], result.scvaByCounterparty().get("A"), 1.0e-6 * scva[0]);
  }

  @Test
  void aSingleNameHedgeCutsTheChargeAndFullSitsBetween() {
    BaCva baCva = new BaCva(PARAMETERS);
    List<BaCva.Exposure> exposures = List.of(
        new BaCva.Exposure(name("A", CreditName.Rating.A, CreditName.Sector.FINANCIAL), 4.0, 30_000_000.0));

    BaCvaResult unhedged = baCva.charge(exposures, List.of());
    BaCvaResult hedged = baCva.charge(exposures,
        List.of(CvaHedge.singleName("A", 20_000_000.0, 4.0, 0.05, 1.0)));

    assertEquals(unhedged.reduced(), hedged.reduced(), 1.0e-9,
        "reduced ignores hedges");
    assertTrue(hedged.full() < unhedged.full(), "a matching CDS hedge must lower the full charge");
    assertTrue(hedged.full() >= PARAMETERS.beta() * hedged.reduced() - 1.0e-6,
        "full is floored by beta * reduced");
    assertTrue(hedged.hedgeBenefit() > 0.0, "hedge benefit is positive");
  }
}
