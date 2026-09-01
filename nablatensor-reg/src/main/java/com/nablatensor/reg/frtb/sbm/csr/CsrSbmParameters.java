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
package com.nablatensor.reg.frtb.sbm.csr;

import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskClassProfile;
import com.nablatensor.risk.RiskFactor;

/**
 * FRTB SA-SBM <b>credit-spread risk (CSR)</b> parameters, from the Basel
 * Framework MAR21. One class serves the three CSR risk classes via the
 * factories:
 *
 * <ul>
 *   <li>{@link #nonSec()} — CSR non-securitisations (MAR21.53–21.62). Risk
 *       weights by the 18 sector/quality buckets, {@code rho_name = 35%}
 *       (indices 80%), {@code rho_tenor = 65%}, {@code rho_basis = 99.90%} for
 *       bond vs CDS.</li>
 *   <li>{@link #securitisation()} — CSR securitisations, non-CTP
 *       (MAR21.63–21.70). <b>Illustrative placeholder tables</b> with the
 *       correct structure — replace the risk-weight array with the MAR21.66
 *       values for the 25 asset-class/seniority/region buckets.</li>
 *   <li>{@link #ctp()} — CSR securitisations, correlation trading portfolio
 *       (MAR21.71–21.78). <b>Illustrative placeholder tables.</b></li>
 * </ul>
 *
 * <p>The across-bucket correlation {@code gamma} here is a <b>simplification</b>
 * of the MAR21.60 matrix: same bucket {@code 1}, same rating tier {@code 50%},
 * different tier {@code 40%}, either bucket is an index bucket {@code 45%},
 * {@code 0%} against the "other sector" bucket. Replace with the published
 * matrix before any real use.
 *
 * <p>Tenor vertices {@code 0.5 1 3 5 10} years. Curvature shock: an absolute
 * parallel shift of the credit-spread curve by the bucket risk weight. Vega risk
 * weight {@code min(0.55*sqrt(LH/10), 1)}, {@code LH = 120}.
 */
public final class CsrSbmParameters implements RiskClassProfile {

  // ---- CSR non-securitisations: risk weight by bucket 1..18 (indicative, MAR21.55) ----
  private static final double[] NON_SEC_RW = {
      Double.NaN,
      0.005, 0.010, 0.050, 0.030, 0.030, 0.020, 0.015, 0.025,   // 1..8 investment grade
      0.020, 0.040, 0.120, 0.070, 0.085, 0.055, 0.050, 0.120,   // 9..16 high yield / non-rated
      0.015, 0.050                                              // 17..18 qualified indices (IG / HY)
  };

  // ---- CSR securitisation (non-CTP): ILLUSTRATIVE placeholder, 25 buckets ----
  private static final double[] SEC_RW = illustrative(25, 0.009, 0.075);

  // ---- CSR securitisation (CTP): ILLUSTRATIVE placeholder, 16 buckets ----
  private static final double[] CTP_RW = illustrative(16, 0.040, 0.130);

  private static final double[] VERTICES = {0.5, 1, 3, 5, 10};
  private static final double TENOR_RHO = 0.65;
  private static final double BASIS_RHO = 0.999;
  private static final double VEGA_LH = 120.0;
  private static final double VEGA_ALPHA = 0.01;

  private final RiskClass riskClass;
  private final double[] rwByBucket;
  private final double nameRho;
  private final double indexNameRho;
  private final boolean hasBasis;

  private CsrSbmParameters(RiskClass riskClass, double[] rwByBucket, double nameRho,
                           double indexNameRho, boolean hasBasis) {
    this.riskClass = riskClass;
    this.rwByBucket = rwByBucket;
    this.nameRho = nameRho;
    this.indexNameRho = indexNameRho;
    this.hasBasis = hasBasis;
  }

  public static CsrSbmParameters nonSec() {
    return new CsrSbmParameters(RiskClass.CSR_NON_SEC, NON_SEC_RW, 0.35, 0.80, true);
  }

  public static CsrSbmParameters securitisation() {
    return new CsrSbmParameters(RiskClass.CSR_SEC, SEC_RW, 0.40, 0.80, false);
  }

  public static CsrSbmParameters ctp() {
    return new CsrSbmParameters(RiskClass.CSR_SEC_CTP, CTP_RW, 0.35, 0.80, true);
  }

  @Override
  public RiskClass riskClass() {
    return riskClass;
  }

  @Override
  public double deltaRiskWeight(RiskFactor k) {
    return rwByBucket[bucket(k)];
  }

  @Override
  public double vegaRiskWeight(RiskFactor k) {
    return Math.min(0.55 * Math.sqrt(VEGA_LH / 10.0), 1.0);
  }

  @Override
  public double curvatureShock(RiskFactor k, double riskFactorLevel) {
    return deltaRiskWeight(k);   // absolute spread shift by the bucket risk weight
  }

  @Override
  public double deltaRho(RiskFactor k, RiskFactor l) {
    if (k.equals(l)) {
      return 1.0;
    }
    if (!k.bucket().equals(l.bucket())) {
      return 0.0;
    }
    double rn = k.csrIssuer().equals(l.csrIssuer()) ? 1.0 : (isIndexBucket(bucket(k)) ? indexNameRho : nameRho);
    double rt = k.tenor() == l.tenor() ? 1.0 : TENOR_RHO;
    double rb = 1.0;
    if (hasBasis && !k.csrCurve().equals(l.csrCurve())) {
      rb = BASIS_RHO;
    }
    return rn * rt * rb;
  }

  @Override
  public double vegaRho(RiskFactor k, RiskFactor l) {
    if (k.equals(l)) {
      return 1.0;
    }
    if (!k.bucket().equals(l.bucket())) {
      return 0.0;
    }
    double rn = k.csrIssuer().equals(l.csrIssuer()) ? 1.0 : (isIndexBucket(bucket(k)) ? indexNameRho : nameRho);
    double x = Math.max(k.tenor(), 1e-9);
    double y = Math.max(l.tenor(), 1e-9);
    double rOpt = Math.exp(-VEGA_ALPHA * Math.abs(x - y) / Math.min(x, y));
    return Math.min(rn * rOpt, 1.0);
  }

  @Override
  public double gamma(String bucketB, String bucketC) {
    if (bucketB.equals(bucketC)) {
      return 1.0;
    }
    int b = Integer.parseInt(bucketB);
    int c = Integer.parseInt(bucketC);
    if (isOtherSectorBucket(b) || isOtherSectorBucket(c)) {
      return 0.0;
    }
    if (isIndexBucket(b) || isIndexBucket(c)) {
      return 0.45;
    }
    return sameRatingTier(b, c) ? 0.50 : 0.40;
  }

  private int bucket(RiskFactor k) {
    return Integer.parseInt(k.bucket());
  }

  private boolean isIndexBucket(int b) {
    return riskClass == RiskClass.CSR_NON_SEC && b >= 17;
  }

  private boolean isOtherSectorBucket(int b) {
    return riskClass == RiskClass.CSR_NON_SEC && b == 16;
  }

  private boolean sameRatingTier(int b, int c) {
    return investmentGrade(b) == investmentGrade(c);
  }

  private boolean investmentGrade(int b) {
    return switch (riskClass) {
      case CSR_NON_SEC -> b <= 8 || b == 17;
      default -> b <= rwByBucket.length / 2;
    };
  }

  private static double[] illustrative(int nBuckets, double lo, double hi) {
    double[] rw = new double[nBuckets + 1];
    rw[0] = Double.NaN;
    for (int i = 1; i <= nBuckets; i++) {
      rw[i] = lo + (hi - lo) * (i - 1) / Math.max(1, nBuckets - 1);
    }
    return rw;
  }

  /** The CSR tenor vertices in years. */
  public static double[] vertices() {
    return VERTICES.clone();
  }
}
