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
package com.nablatensor.risk;

/** The FRTB / SIMM risk classes. Phase-2 slice implements EQUITY end to end. */
public enum RiskClass {
  /** General interest-rate risk: yield-curve, inflation and cross-currency-basis factors. */
  GIRR,
  /** Credit-spread risk on non-securitisation positions (bonds, single-name / index CDS). */
  CSR_NON_SEC,
  /** Credit-spread risk on securitisations outside the correlation-trading portfolio. */
  CSR_SEC,
  /** Credit-spread risk on securitisations in the correlation-trading portfolio (n-th-to-default, bespoke tranches). */
  CSR_SEC_CTP,
  /** Equity risk: spot and repo-rate factors. */
  EQUITY,
  /** Commodity risk: forward-price factors keyed by commodity and delivery location. */
  COMMODITY,
  /** Foreign-exchange risk: one factor per currency pair. */
  FX
}
