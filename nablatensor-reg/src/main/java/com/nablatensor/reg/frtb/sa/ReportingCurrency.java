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

import java.util.Map;

/**
 * The bank's reporting currency and the rates to convert other-currency amounts
 * into it. FRTB SA is computed in the reporting currency (MAR21); SBM
 * sensitivities are assumed already expressed in it, while DRC / RRAO position
 * notionals may be supplied per currency and converted here.
 *
 * @param code           ISO code of the reporting currency (e.g. {@code "EUR"})
 * @param ratesToReporting units of reporting currency per one unit of the keyed currency; the
 *                         reporting currency itself need not appear (rate 1)
 */
public record ReportingCurrency(String code, Map<String, Double> ratesToReporting) {

  public ReportingCurrency {
    ratesToReporting = Map.copyOf(ratesToReporting);
  }

  /** A reporting currency with no conversion table — every amount is assumed already converted. */
  public static ReportingCurrency of(String code) {
    return new ReportingCurrency(code, Map.of());
  }

  public static ReportingCurrency of(String code, Map<String, Double> ratesToReporting) {
    return new ReportingCurrency(code, ratesToReporting);
  }

  /** {@code amount} (in {@code currency}) expressed in the reporting currency. */
  public double convert(double amount, String currency) {
    if (currency.equals(code)) {
      return amount;
    }
    Double rate = ratesToReporting.get(currency);
    if (rate == null) {
      throw new IllegalArgumentException("no rate from " + currency + " to reporting currency " + code);
    }
    return amount * rate;
  }
}
