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

import com.nablatensor.reg.frtb.sbm.SbmCharge;
import com.nablatensor.reg.frtb.sbm.csr.CsrSbmParameters;
import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.RiskFactor.CsrCurve;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * LESSON 10 — CSR (credit-spread risk): the within-bucket correlation is a
 * <b>product of three factors</b> (MAR21):
 *
 * <pre>
 *   rho = rho_name  *  rho_tenor  *  rho_basis
 *          |             |             |
 *          35%           65%           99.9%   (bond curve vs CDS curve, same name)
 *        (1 if same issuer) (1 if same tenor) (1 if same curve type)
 * </pre>
 *
 * Consequence: a bond hedged one-for-one with a CDS on the <b>same name, same
 * tenor</b> is not a capital-free hedge — {@code rho_basis = 0.999 < 1} leaves a
 * residual "basis risk" charge of roughly {@code sqrt(2*(1 - 0.999)) ~ 4.5%} of
 * one leg.
 */
class Lesson10_CsrBondCdsBasisTest {

  private static final CsrSbmParameters CSR = CsrSbmParameters.nonSec();

  @Test
  void sameNameSameTenor_bondVsCds_correlateAt0_999() {
    double rho = CSR.deltaRho(
        RiskFactor.csrDelta("3", "ACME", CsrCurve.BOND, 5),
        RiskFactor.csrDelta("3", "ACME", CsrCurve.CDS, 5));
    assertEquals(0.999, rho, 1e-12);
  }

  @Test
  void differentIssuersInTheSameBucket_correlateAt35Percent() {
    double rho = CSR.deltaRho(
        RiskFactor.csrDelta("3", "ACME", CsrCurve.BOND, 5),
        RiskFactor.csrDelta("3", "GLOBEX", CsrCurve.BOND, 5));
    assertEquals(0.35, rho, 1e-12);
  }

  @Test
  void aNotionalPerfectBondCdsHedge_stillAttractsBasisRiskCapital() {
    // bucket 3 risk weight is 5% ; choose sensitivities so |WS| = 100 on each leg
    Sensitivities book = Sensitivities.builder()
        .add(RiskFactor.csrDelta("3", "ACME", CsrCurve.BOND, 5), 2_000.0)    // WS = +100
        .add(RiskFactor.csrDelta("3", "ACME", CsrCurve.CDS, 5), -2_000.0)    // WS = -100
        .build();

    double medium = SbmCharge.of(CSR).compute(book, List.of())
        .perScenario().get(CorrelationScenario.MEDIUM);

    // K_b = sqrt(100^2 + 100^2 + 2*0.999*100*(-100)) = sqrt(20000 * (1 - 0.999)) = sqrt(20)
    assertEquals(Math.sqrt(20.0), medium, 1e-6);
    assertTrue(medium > 0.0, "the hedge is notional-flat but not capital-flat");
  }
}
