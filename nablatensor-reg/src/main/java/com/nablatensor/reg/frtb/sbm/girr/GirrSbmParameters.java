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
package com.nablatensor.reg.frtb.sbm.girr;

import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskClassProfile;
import com.nablatensor.risk.RiskFactor;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * FRTB SA-SBM <b>GIRR</b> (general interest-rate risk) parameters, from the
 * Basel Framework MAR21. Values as published for the risk-free yield curve, plus
 * inflation and cross-currency basis:
 *
 * <ul>
 *   <li>10 tenor vertices {@code 0.25 0.5 1 2 3 5 10 15 20 30} years;</li>
 *   <li>delta risk weights by vertex {@code 1.7 1.7 1.6 1.3 1.2 1.1 1.1 1.1 1.1
 *       1.1} percent; inflation and cross-currency basis {@code 1.6%} flat;</li>
 *   <li>for the specified liquid currencies (EUR, USD, GBP, AUD, JPY, SEK, CAD
 *       and the bank's domestic currency) every delta risk weight is divided by
 *       {@code sqrt(2)};</li>
 *   <li>within-bucket correlation: same curve, different vertices
 *       {@code max(exp(-theta*|Tk-Tl|/min(Tk,Tl)), 40%)}, {@code theta = 3%};
 *       different curves multiply by {@code 0.999}; inflation vs a yield vertex
 *       {@code 40%}; cross-currency basis vs anything {@code 0%};</li>
 *   <li>across-bucket (currency) correlation {@code gamma = 50%};</li>
 *   <li>curvature shock: an <b>absolute</b> parallel shift of the whole curve by
 *       the highest GIRR delta risk weight (no {@code sqrt(2)} relief);</li>
 *   <li>vega risk weight {@code min(0.55*sqrt(LH/10), 1)}, {@code LH_GIRR = 60}
 *       (so effectively 100%); vega correlation from option- and underlying-
 *       maturity distance.</li>
 * </ul>
 *
 * <p>This is a data table; check it against your regulator's current rulebook
 * (the EU liquid-currency list and the targeted-relief multipliers differ).
 */
public final class GirrSbmParameters implements RiskClassProfile {

  private static final double[] VERTICES = {0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30};
  private static final double[] VERTEX_RW = {
      0.017, 0.017, 0.016, 0.013, 0.012, 0.011, 0.011, 0.011, 0.011, 0.011
  };
  private static final double INFLATION_RW = 0.016;
  private static final double XCCY_BASIS_RW = 0.016;
  private static final double THETA = 0.03;
  private static final double GAMMA = 0.50;
  private static final double DIFFERENT_CURVE = 0.999;
  private static final double INFLATION_VS_CURVE = 0.40;
  private static final double CORR_FLOOR = 0.40;
  private static final double VEGA_LH = 60.0;
  private static final double VEGA_ALPHA = 0.01;

  /** The Basel-specified liquid currencies whose GIRR delta risk weights are divided by sqrt(2). */
  public static final Set<String> BASEL_LIQUID_CURRENCIES =
      Set.of("EUR", "USD", "GBP", "AUD", "JPY", "SEK", "CAD");

  private final Set<String> reducedRwCurrencies;

  private GirrSbmParameters(Set<String> reducedRwCurrencies) {
    this.reducedRwCurrencies = Set.copyOf(reducedRwCurrencies);
  }

  /** The Basel default: the seven specified liquid currencies get the sqrt(2) relief. */
  public static GirrSbmParameters baselDefault() {
    return new GirrSbmParameters(BASEL_LIQUID_CURRENCIES);
  }

  /** The Basel default plus a domestic currency that also gets the sqrt(2) relief. */
  public static GirrSbmParameters withDomestic(String domesticCurrency) {
    Set<String> s = new LinkedHashSet<>(BASEL_LIQUID_CURRENCIES);
    s.add(domesticCurrency);
    return new GirrSbmParameters(s);
  }

  @Override
  public RiskClass riskClass() {
    return RiskClass.GIRR;
  }

  @Override
  public double deltaRiskWeight(RiskFactor k) {
    double base = switch (k.name()) {
      case "INFL" -> INFLATION_RW;
      case "XCCY" -> XCCY_BASIS_RW;
      default -> VERTEX_RW[vertexIndex(k.tenor())];
    };
    return reducedRwCurrencies.contains(k.bucket()) ? base / Math.sqrt(2.0) : base;
  }

  @Override
  public double vegaRiskWeight(RiskFactor k) {
    return Math.min(0.55 * Math.sqrt(VEGA_LH / 10.0), 1.0);
  }

  @Override
  public double curvatureShock(RiskFactor k, double riskFactorLevel) {
    // Absolute parallel shift by the highest GIRR delta risk weight; no sqrt(2) relief.
    double hi = 0.0;
    for (double rw : VERTEX_RW) {
      hi = Math.max(hi, rw);
    }
    return Math.max(hi, Math.max(INFLATION_RW, XCCY_BASIS_RW));
  }

  @Override
  public double deltaRho(RiskFactor k, RiskFactor l) {
    if (k.equals(l)) {
      return 1.0;
    }
    if (!k.bucket().equals(l.bucket())) {
      return 0.0;   // different currency -> handled by gamma
    }
    boolean kX = k.name().equals("XCCY");
    boolean lX = l.name().equals("XCCY");
    if (kX || lX) {
      return 0.0;
    }
    boolean kI = k.name().equals("INFL");
    boolean lI = l.name().equals("INFL");
    if (kI && lI) {
      return 1.0;
    }
    if (kI != lI) {
      return INFLATION_VS_CURVE;
    }
    // both are yield-curve vertices
    double tk = Math.max(k.tenor(), 1e-9);
    double tl = Math.max(l.tenor(), 1e-9);
    double rho = Math.max(Math.exp(-THETA * Math.abs(tk - tl) / Math.min(tk, tl)), CORR_FLOOR);
    if (!k.name().equals(l.name())) {
      rho *= DIFFERENT_CURVE;
    }
    return rho;
  }

  @Override
  public double vegaRho(RiskFactor k, RiskFactor l) {
    if (k.equals(l)) {
      return 1.0;
    }
    if (!k.bucket().equals(l.bucket())) {
      return 0.0;
    }
    double rhoOpt = expDecay(k.tenor(), l.tenor());
    double rhoUnd = expDecay(k.tenor2(), l.tenor2());
    return Math.min(rhoOpt * rhoUnd, 1.0);
  }

  @Override
  public double gamma(String bucketB, String bucketC) {
    return bucketB.equals(bucketC) ? 1.0 : GAMMA;
  }

  private static double expDecay(double a, double b) {
    double x = Math.max(a, 1e-9);
    double y = Math.max(b, 1e-9);
    return Math.exp(-VEGA_ALPHA * Math.abs(x - y) / Math.min(x, y));
  }

  private static int vertexIndex(double tenor) {
    int best = 0;
    double bestDiff = Double.MAX_VALUE;
    for (int i = 0; i < VERTICES.length; i++) {
      double diff = Math.abs(VERTICES[i] - tenor);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = i;
      }
    }
    return best;
  }

  /** The 10 GIRR tenor vertices in years. */
  public static double[] vertices() {
    return VERTICES.clone();
  }
}
