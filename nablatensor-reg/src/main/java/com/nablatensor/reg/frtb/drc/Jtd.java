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
package com.nablatensor.reg.frtb.drc;

import com.nablatensor.reg.frtb.drc.DrcParameters.Seniority;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Jump-to-default arithmetic (MAR22.8–22.22):
 *
 * <pre>{@code
 * gross JTD  = ( LGD * notional + (marketValue - notional) ) * maturityScale
 * maturityScale = 1                          for equity (1-year floor)
 *               = clamp(maturityYears, 0.25, 1)   otherwise
 * net JTD(obligor) = sum of the same obligor's signed gross JTD
 * }</pre>
 *
 * <p>Simplification: same-obligor longs and shorts net unconditionally. MAR22.21
 * only permits the short to offset a long of the same or lower seniority; wire
 * that constraint in before real use.
 */
public final class Jtd {

  private Jtd() {
  }

  /** Signed, maturity-scaled gross JTD for one position. */
  public static double gross(DefaultRiskPosition p) {
    double raw = p.seniority().lgd() * p.notional() + (p.marketValue() - p.notional());
    return raw * maturityScale(p);
  }

  /** Net JTD per obligor: the sum of that obligor's signed gross JTD, in first-seen order. */
  public static Map<String, Double> netByObligor(List<DefaultRiskPosition> positions) {
    Map<String, Double> net = new LinkedHashMap<>();
    for (DefaultRiskPosition p : positions) {
      net.merge(p.obligor(), gross(p), Double::sum);
    }
    return net;
  }

  private static double maturityScale(DefaultRiskPosition p) {
    if (p.seniority() == Seniority.EQUITY) {
      return 1.0;
    }
    return Math.min(1.0, Math.max(p.maturityYears(), 0.25));
  }
}
