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

import java.util.Map;

/**
 * The BA-CVA capital charge, reduced and full.
 *
 * @param reduced       {@code K_reduced} — the systematic / idiosyncratic blend with no hedges
 * @param full          {@code K_full = beta*K_reduced + (1-beta)*K_hedged}
 * @param scvaByCounterparty  the per-counterparty {@code SCVA_c} contributions
 * @param hedgeBenefit  {@code reduced - full}, the recognised CDS hedge offset
 */
public record BaCvaResult(double reduced, double full,
                          Map<String, Double> scvaByCounterparty, double hedgeBenefit) {

  public BaCvaResult {
    scvaByCounterparty = Map.copyOf(scvaByCounterparty);
  }
}
