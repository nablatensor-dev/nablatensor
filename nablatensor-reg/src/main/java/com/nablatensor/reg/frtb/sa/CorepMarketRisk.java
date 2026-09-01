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
package com.nablatensor.reg.frtb.sa;

import com.nablatensor.risk.CorrelationScenario;
import com.nablatensor.risk.RiskClass;
import java.util.Map;

/**
 * A COREP-shaped view of an FRTB SA result — the rows of the EU {@code C 90.xx}
 * market-risk templates, as plain data. No XBRL binding: this is a structure to
 * map into a reporting tool, not a submission artefact.
 *
 * @param reportingCurrency ISO code the figures are expressed in
 * @param parameterSet      name of the {@link FrtbParameterSet} used
 * @param perRiskClass      SBM breakdown by risk class
 * @param sbmTotal          Sum over risk classes of max(high, medium, low)
 * @param drc               default risk charge
 * @param rrao              residual risk add-on
 * @param total             {@code sbmTotal + drc + rrao}
 */
public record CorepMarketRisk(String reportingCurrency, String parameterSet,
                              Map<RiskClass, Row> perRiskClass,
                              double sbmTotal, double drc, double rrao, double total) {

  /**
   * One SBM risk-class row.
   *
   * @param delta            delta charge at the binding scenario
   * @param vega             vega charge at the binding scenario
   * @param curvature        curvature charge at the binding scenario
   * @param classCharge      {@code delta + vega + curvature} = max over the three scenarios
   * @param bindingScenario  which correlation scenario bound
   */
  public record Row(double delta, double vega, double curvature, double classCharge,
                    CorrelationScenario bindingScenario) {
  }
}
