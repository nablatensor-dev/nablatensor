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
 * LESSON 8 — the GIRR "liquid currency" relief. For the specified currencies
 * (EUR, USD, GBP, AUD, JPY, SEK, CAD, plus the bank's domestic currency) every
 * GIRR delta risk weight is <b>divided by sqrt(2)</b> — a recognition that those
 * curves are deeper and less volatile.
 *
 * <pre>
 *   same 10y PV01 of 10,000 in each currency:
 *
 *   EUR (liquid)   RW = 1.1% / sqrt(2) = 0.778%   ->  WS =  77.8
 *   PLN (other)    RW = 1.1%                       ->  WS = 110.0
 *
 *   identical interest-rate risk, ~41% more capital in the non-liquid currency
 * </pre>
 */
class Lesson08_GirrLiquidCurrencyReliefTest {

  @Test
  void liquidCurrencyRiskWeightIsTheBaseWeightOverSqrt2() {
    GirrSbmParameters girr = GirrSbmParameters.baselDefault();
    double eur = girr.deltaRiskWeight(RiskFactor.girrDelta("EUR", "OIS", 10));
    double pln = girr.deltaRiskWeight(RiskFactor.girrDelta("PLN", "OIS", 10));

    assertEquals(pln / Math.sqrt(2.0), eur, 1e-12);
    assertTrue(pln > eur);
  }

  @Test
  void aDomesticCurrencyCanBeAddedToTheReliefList() {
    GirrSbmParameters withPln = GirrSbmParameters.withDomestic("PLN");
    double plnRelieved = withPln.deltaRiskWeight(RiskFactor.girrDelta("PLN", "OIS", 10));
    double plnPlain = GirrSbmParameters.baselDefault().deltaRiskWeight(RiskFactor.girrDelta("PLN", "OIS", 10));

    assertEquals(plnPlain / Math.sqrt(2.0), plnRelieved, 1e-12);
  }

  @Test
  void curvatureShockIgnoresTheRelief() {
    // MAR21: the curvature shock is the highest *un-relieved* GIRR delta risk weight
    GirrSbmParameters girr = GirrSbmParameters.baselDefault();
    double eurShock = girr.curvatureShock(RiskFactor.girrDelta("EUR", "OIS", 10), 0.0);
    double plnShock = girr.curvatureShock(RiskFactor.girrDelta("PLN", "OIS", 10), 0.0);
    assertEquals(eurShock, plnShock, 1e-12);
    assertEquals(0.017, eurShock, 1e-12);   // the 0.25y / 0.5y weight, the largest in the table
  }
}
