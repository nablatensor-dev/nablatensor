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
package com.nablatensor.reg.tour;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.reg.frtb.sbm.girr.GirrSbmParameters;
import com.nablatensor.risk.RiskFactor;
import org.junit.jupiter.api.Test;

/**
 * LESSON 7 — GIRR (general interest-rate risk): the yield curve is sampled at
 * <b>10 fixed tenor vertices</b> and the correlation between two vertices decays
 * with their separation (MAR21):
 *
 * <pre>
 *   rho(T_k, T_l) = max( exp( -theta * |T_k - T_l| / min(T_k, T_l) ) , 40% ),   theta = 3%
 *
 *   rho
 *   1.00 *·.
 *        |  ·..
 *   0.80 |     ·..
 *        |        ·...
 *   0.60 |            ·.....
 *        |                  ·........
 *   0.40 |·······················___________  <- 40% floor: far-apart tenors never decorrelate below this
 *        +----+----+----+----+----+----+----> |T_k - T_l| (relative)
 * </pre>
 *
 * A separate 0.999 multiplier applies between two <b>different curves</b> of the
 * same currency (e.g. OIS vs 3M).
 */
class Lesson07_GirrTenorVerticesTest {

  private static final GirrSbmParameters GIRR = GirrSbmParameters.baselDefault();

  @Test
  void thereAreTenVertices_from3MonthsTo30Years() {
    double[] v = GirrSbmParameters.vertices();
    assertEquals(10, v.length);
    assertEquals(0.25, v[0], 1e-12);
    assertEquals(30.0, v[9], 1e-12);
  }

  @Test
  void nearbyTenorsAreHighlyCorrelated() {
    double rho = GIRR.deltaRho(
        RiskFactor.girrDelta("EUR", "OIS", 2), RiskFactor.girrDelta("EUR", "OIS", 5));
    assertEquals(Math.exp(-0.03 * 3.0 / 2.0), rho, 1e-9);   // ~0.956
  }

  @Test
  void farApartTenorsHitThe40PercentFloor() {
    double rho = GIRR.deltaRho(
        RiskFactor.girrDelta("EUR", "OIS", 0.25), RiskFactor.girrDelta("EUR", "OIS", 30));
    assertEquals(0.40, rho, 1e-12);   // exp(...) would be ~0.03, floored to 0.40
  }

  @Test
  void sameVertexDifferentCurve_carriesThe0_999Multiplier() {
    double rho = GIRR.deltaRho(
        RiskFactor.girrDelta("EUR", "OIS", 5), RiskFactor.girrDelta("EUR", "3M", 5));
    assertEquals(0.999, rho, 1e-12);   // exp(0) = 1, times the 0.999 cross-curve factor
  }

  @Test
  void correlationDecreasesWithSeparation() {
    double near = GIRR.deltaRho(RiskFactor.girrDelta("EUR", "OIS", 2), RiskFactor.girrDelta("EUR", "OIS", 3));
    double far = GIRR.deltaRho(RiskFactor.girrDelta("EUR", "OIS", 2), RiskFactor.girrDelta("EUR", "OIS", 20));
    assertTrue(near > far);
  }
}
