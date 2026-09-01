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

import com.nablatensor.reg.frtb.drc.DrcParameters.DrcBucket;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * FRTB SA <b>Default Risk Charge</b> for non-securitisations (MAR22.23–22.24).
 *
 * <pre>{@code
 * per bucket b:
 *   HBR_b = Sum netJTD_long / ( Sum netJTD_long + Sum |netJTD_short| )
 *   DRC_b = max( Sum_i RW_i * netJTD_long_i  -  HBR_b * Sum_i RW_i * |netJTD_short_i| , 0 )
 * DRC = Sum_b DRC_b          (no diversification across the three buckets)
 * }</pre>
 *
 * <p>Calculators, not sign-off.
 */
public final class DefaultRiskCharge {

  /** Per-bucket intermediates plus the total. */
  public record Result(double total, Map<DrcBucket, Double> perBucket,
                       Map<DrcBucket, Double> hedgeBenefitRatio) {
  }

  private final List<DefaultRiskPosition> positions;

  private DefaultRiskCharge(List<DefaultRiskPosition> positions) {
    this.positions = List.copyOf(positions);
  }

  public static DefaultRiskCharge of(List<DefaultRiskPosition> positions) {
    return new DefaultRiskCharge(positions);
  }

  public Result compute() {
    // obligor -> (bucket, quality) ; obligor -> net JTD
    Map<String, DefaultRiskPosition> ref = new LinkedHashMap<>();
    for (DefaultRiskPosition p : positions) {
      ref.putIfAbsent(p.obligor(), p);
    }
    Map<String, Double> net = Jtd.netByObligor(positions);

    Map<DrcBucket, double[]> acc = new EnumMap<>(DrcBucket.class);   // [wLong, wShortAbs, sLong, sShortAbs]
    for (var e : net.entrySet()) {
      DefaultRiskPosition r = ref.get(e.getKey());
      double jtd = e.getValue();
      double rw = r.quality().riskWeight();
      double[] a = acc.computeIfAbsent(r.bucket(), b -> new double[4]);
      if (jtd >= 0.0) {
        a[0] += rw * jtd;
        a[2] += jtd;
      } else {
        a[1] += rw * (-jtd);
        a[3] += -jtd;
      }
    }

    Map<DrcBucket, Double> perBucket = new EnumMap<>(DrcBucket.class);
    Map<DrcBucket, Double> hbrMap = new EnumMap<>(DrcBucket.class);
    double total = 0.0;
    for (var e : acc.entrySet()) {
      double[] a = e.getValue();
      double denom = a[2] + a[3];
      double hbr = denom == 0.0 ? 0.0 : a[2] / denom;
      double drcB = Math.max(a[0] - hbr * a[1], 0.0);
      perBucket.put(e.getKey(), drcB);
      hbrMap.put(e.getKey(), hbr);
      total += drcB;
    }
    return new Result(total, perBucket, hbrMap);
  }

  /** The DRC total for a set of positions. */
  public static double total(List<DefaultRiskPosition> positions) {
    return of(positions).compute().total();
  }
}
