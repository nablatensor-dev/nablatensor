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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The two-level correlation aggregation at the heart of FRTB SA-SBM and ISDA
 * SIMM: weight each sensitivity, aggregate within a bucket with a correlation
 * matrix, aggregate the bucket totals with a second matrix.
 *
 * <pre>{@code
 * WS_k  = RW_k * s_k                 (delta / vega;  curvature feeds CVR_k directly)
 * K_b   = sqrt(max(0, sum_k D(WS_k) + sum_{k!=l} R(rho_kl) WS_k WS_l psi_kl))
 * S_b   = clamp(sum_k WS_k, -K_b, K_b)
 * total = sqrt(max(0, sum_b K_b^2 + sum_{b!=c} R(gamma_bc) S_b S_c psi_bc))
 * }</pre>
 *
 * where for <b>delta / vega</b> {@code D(w)=w^2}, {@code R(c)=c}, {@code psi=1};
 * for <b>curvature</b> {@code D(w)=max(w,0)^2}, {@code R(c)=c^2}, and
 * {@code psi(a,b)=0} iff both are negative.
 */
public final class NestedAggregation {

  @FunctionalInterface
  public interface RiskWeight {
    double weight(RiskFactor factor);
  }

  @FunctionalInterface
  public interface WithinBucketCorrelation {
    double rho(RiskFactor k, RiskFactor l);
  }

  @FunctionalInterface
  public interface AcrossBucketCorrelation {
    double gamma(String bucketB, String bucketC);
  }

  /** Per-bucket intermediates plus the aggregate. */
  public record Result(double total, Map<String, Double> kb, Map<String, Double> sb) {}

  private final RiskWeight rw;
  private final WithinBucketCorrelation rho;
  private final AcrossBucketCorrelation gamma;
  private final boolean curvature;
  private final ConcentrationFactor concentration;   // SIMM CR_k; identity for FRTB
  private final boolean applyFkl;                     // SIMM within-bucket concentration correction

  /** SIMM concentration risk factor {@code CR_k}; FRTB uses {@link #NONE}. */
  @FunctionalInterface
  public interface ConcentrationFactor {
    double cr(RiskFactor factor, double sensitivity);

    ConcentrationFactor NONE = (f, s) -> 1.0;
  }

  private NestedAggregation(RiskWeight rw, WithinBucketCorrelation rho, AcrossBucketCorrelation gamma,
                            boolean curvature, ConcentrationFactor concentration, boolean applyFkl) {
    this.rw = rw;
    this.rho = rho;
    this.gamma = gamma;
    this.curvature = curvature;
    this.concentration = concentration;
    this.applyFkl = applyFkl;
  }

  public static NestedAggregation delta(RiskWeight rw, WithinBucketCorrelation rho, AcrossBucketCorrelation gamma) {
    return new NestedAggregation(rw, rho, gamma, false, ConcentrationFactor.NONE, false);
  }

  public static NestedAggregation curvature(WithinBucketCorrelation rho, AcrossBucketCorrelation gamma) {
    return new NestedAggregation((f) -> 1.0, rho, gamma, true, ConcentrationFactor.NONE, false);
  }

  /**
   * Adds the SIMM concentration risk factor {@code CR_k}: it scales every
   * weighted sensitivity and multiplies the within-bucket cross terms by
   * {@code f_kl = min(CR_k,CR_l)/max(CR_k,CR_l)}. The across-bucket {@code g_bc}
   * correction is treated as 1 in this slice.
   */
  public NestedAggregation withConcentration(ConcentrationFactor cr) {
    return new NestedAggregation(rw, rho, gamma, curvature, cr, true);
  }

  /**
   * @param bucketed sensitivities already restricted to one risk class and one measure
   */
  public Result aggregate(Sensitivities bucketed) {
    Map<String, List<RiskFactor>> byBucket = new LinkedHashMap<>();
    Map<RiskFactor, Double> ws = new LinkedHashMap<>();
    Map<RiskFactor, Double> cr = new LinkedHashMap<>();
    for (var e : bucketed.asMap().entrySet()) {
      RiskFactor f = e.getKey();
      double s = e.getValue();
      double c = concentration.cr(f, s);
      byBucket.computeIfAbsent(f.bucket(), b -> new ArrayList<>()).add(f);
      ws.put(f, rw.weight(f) * s * c);
      cr.put(f, c);
    }

    Map<String, Double> kb = new LinkedHashMap<>();
    Map<String, Double> sb = new LinkedHashMap<>();
    for (var e : byBucket.entrySet()) {
      List<RiskFactor> fs = e.getValue();
      double sum = 0.0;
      double sumWs = 0.0;
      for (int i = 0; i < fs.size(); i++) {
        double wi = ws.get(fs.get(i));
        sumWs += wi;
        sum += curvature ? Math.pow(Math.max(wi, 0.0), 2) : wi * wi;
        for (int j = 0; j < fs.size(); j++) {
          if (i == j) {
            continue;
          }
          double wj = ws.get(fs.get(j));
          double corr = rho.rho(fs.get(i), fs.get(j));
          double f = applyFkl
              ? Math.min(cr.get(fs.get(i)), cr.get(fs.get(j))) / Math.max(cr.get(fs.get(i)), cr.get(fs.get(j)))
              : 1.0;
          sum += (curvature ? corr * corr : corr) * f * wi * wj * psi(wi, wj);
        }
      }
      double k = Math.sqrt(Math.max(0.0, sum));
      kb.put(e.getKey(), k);
      sb.put(e.getKey(), Math.max(-k, Math.min(sumWs, k)));
    }

    List<String> bs = new ArrayList<>(kb.keySet());
    double total = 0.0;
    for (int i = 0; i < bs.size(); i++) {
      total += kb.get(bs.get(i)) * kb.get(bs.get(i));
      for (int j = 0; j < bs.size(); j++) {
        if (i == j) {
          continue;
        }
        double g = gamma.gamma(bs.get(i), bs.get(j));
        double si = sb.get(bs.get(i));
        double sj = sb.get(bs.get(j));
        total += (curvature ? g * g : g) * si * sj * psi(si, sj);
      }
    }
    return new Result(Math.sqrt(Math.max(0.0, total)), kb, sb);
  }

  private double psi(double a, double b) {
    return curvature && a < 0.0 && b < 0.0 ? 0.0 : 1.0;
  }
}
