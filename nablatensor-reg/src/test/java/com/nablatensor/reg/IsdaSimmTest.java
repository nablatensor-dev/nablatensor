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

import com.nablatensor.reg.simm.IsdaSimm;
import com.nablatensor.reg.simm.SimmEquityParameters;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import org.junit.jupiter.api.Test;

/**
 * ISDA SIMM equity aggregation, reconciled to a hand-worked example: the delta
 * margin with the concentration risk factor {@code CR_k} and the
 * {@code f_kl = min/max} within-bucket correction, computed fresh here, must
 * match {@link IsdaSimm}. (Parameter values are illustrative — see
 * {@link SimmEquityParameters}.)
 */
class IsdaSimmTest {

  private static RiskFactor d(String b, String n) {
    return RiskFactor.equityDelta(b, n);
  }

  @Test
  void deltaMarginReconcilesToTheHandCalc() {
    // two large names in bucket 5, one in bucket 6; sizes chosen to trip the concentration threshold
    RiskFactor a = d("5", "ACME");
    RiskFactor b = d("5", "GLOBEX");
    RiskFactor c = d("6", "INITECH");
    double sa = 40_000_000;
    double sb = -12_000_000;
    double sc = 30_000_000;

    Sensitivities book = Sensitivities.builder().add(a, sa).add(b, sb).add(c, sc).build();

    double crA = cr(a, sa);
    double crB = cr(b, sb);
    double crC = cr(c, sc);
    double wsA = SimmEquityParameters.deltaRiskWeight(a) * sa * crA;
    double wsB = SimmEquityParameters.deltaRiskWeight(b) * sb * crB;
    double wsC = SimmEquityParameters.deltaRiskWeight(c) * sc * crC;

    double rho5 = SimmEquityParameters.withinBucketRho(a, b);
    double fAB = Math.min(crA, crB) / Math.max(crA, crB);
    double k5 = Math.sqrt(wsA * wsA + wsB * wsB + 2 * rho5 * fAB * wsA * wsB);
    double k6 = Math.abs(wsC);
    double s5 = Math.max(-k5, Math.min(wsA + wsB, k5));
    double s6 = Math.max(-k6, Math.min(wsC, k6));
    double gamma = SimmEquityParameters.acrossBucketGamma("5", "6");
    double expected = Math.sqrt(k5 * k5 + k6 * k6 + 2 * gamma * s5 * s6);

    IsdaSimm.Result r = IsdaSimm.equity(book);
    assertEquals(expected, r.deltaMargin(), 1e-6 * (1 + expected), "SIMM equity delta margin");
    assertTrue(r.total() >= r.deltaMargin(), "total includes the (zero here) vega margin");
    assertTrue(crA > 1.0, "large position trips the concentration threshold");
  }

  private static double cr(RiskFactor f, double s) {
    return Math.max(1.0, Math.sqrt(Math.abs(s) / SimmEquityParameters.deltaConcentrationThreshold(f)));
  }
}
