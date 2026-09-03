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
import com.nablatensor.risk.RiskClass;
import java.util.EnumMap;
import java.util.Map;

/**
 * The SA-CVA capital charge and its build-up: per risk type and per correlation
 * scenario, with the binding scenario and the {@code m_CVA}-scaled total.
 *
 * @param perScenario  total charge under each of the three correlation scenarios
 * @param byRiskType   per-risk-class charge under the binding scenario
 * @param selected     the binding (largest) correlation scenario
 * @param total        {@code m_CVA * max_scenario(...)}
 */
public record SaCvaResult(Map<CorrelationScenario, Double> perScenario,
                          Map<RiskClass, Double> byRiskType,
                          CorrelationScenario selected, double total) {

  public SaCvaResult {
    perScenario = new EnumMap<>(perScenario);
    byRiskType = new EnumMap<>(byRiskType);
  }
}
