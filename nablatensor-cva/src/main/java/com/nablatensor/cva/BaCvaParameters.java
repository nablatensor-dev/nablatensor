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

/**
 * Parameters for the BA-CVA capital charge (Basel MAR50.1-50.8; CRR3 Art. 384):
 * the supervisory correlation {@code rho = 0.5}, the {@code beta = 0.25} weight
 * on the unhedged charge in the full version, the {@code alpha = 1.4} that
 * converts effective EPE to EAD, and the counterparty risk-weight table by
 * rating and sector.
 *
 * <p>The risk weights are <em>indicative</em>, transcribed from the MAR50.5
 * table shape for a runnable demo; verify against the current rulebook.
 */
public record BaCvaParameters(double rho, double beta, double alpha) {

  public static BaCvaParameters standard() {
    return new BaCvaParameters(0.5, 0.25, 1.4);
  }

  /** Counterparty risk weight {@code RW_c} by rating and sector (MAR50.5, indicative). */
  public double riskWeight(CreditName.Rating rating, CreditName.Sector sector) {
    boolean investmentGrade = switch (rating) {
      case AAA, AA, A, BBB -> true;
      default -> false;
    };
    double base = switch (sector) {
      case SOVEREIGN -> investmentGrade ? 0.005 : 0.020;
      case LOCAL_GOVERNMENT -> investmentGrade ? 0.010 : 0.040;
      case FINANCIAL -> investmentGrade ? 0.050 : 0.120;
      case CORPORATE, TECH -> investmentGrade ? 0.030 : 0.080;
      case CONSUMER -> investmentGrade ? 0.030 : 0.070;
      case OTHER -> investmentGrade ? 0.035 : 0.120;
    };
    return switch (rating) {
      case AAA -> base * 0.7;
      case AA -> base * 0.85;
      case A -> base;
      case BBB -> base * 1.15;
      case BB -> base;
      case B -> base * 1.5;
      case CCC -> base * 3.0;
      case UNRATED -> base * 1.25;
    };
  }

  /** Supervisory discount factor {@code (1 - e^{-0.05 M}) / (0.05 M)}. */
  public double supervisoryDiscount(double effectiveMaturityYears) {
    double x = 0.05 * effectiveMaturityYears;
    return x <= 0.0 ? 1.0 : (1.0 - Math.exp(-x)) / x;
  }
}
