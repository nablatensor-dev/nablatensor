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
 * A variation-margin CSA: a threshold below which no collateral is called, a
 * minimum transfer amount, an independent amount held, and a margin period of
 * risk. {@link ExposureSimulation} applies it on the tape — collateral posted
 * against the netting-set value as of {@code marginPeriodOfRiskDays} earlier is
 * subtracted from the current value, so the collateralised exposure is
 * {@code max(V(t) - C(t - MPoR) - independentAmount, 0)}.
 *
 * @param threshold             unsecured threshold in reporting currency ({@code +inf} = uncollateralised)
 * @param minimumTransfer       minimum transfer amount
 * @param independentAmount     independent amount / initial margin held, reduces exposure
 * @param marginPeriodOfRiskDays close-out period in calendar days
 */
public record CollateralAgreement(double threshold, double minimumTransfer,
                                  double independentAmount, double marginPeriodOfRiskDays) {

  public CollateralAgreement {
    if (!(threshold >= 0.0) || !(minimumTransfer >= 0.0) || !(independentAmount >= 0.0)
        || !(marginPeriodOfRiskDays >= 0.0)) {
      throw new IllegalArgumentException("all CSA terms must be >= 0");
    }
  }

  /** No collateral: an infinite threshold. */
  public static CollateralAgreement uncollateralised() {
    return new CollateralAgreement(Double.POSITIVE_INFINITY, 0.0, 0.0, 0.0);
  }

  /** A daily-margined CSA with a zero threshold and a ten-day MPoR. */
  public static CollateralAgreement dailyMargined(double independentAmount) {
    return new CollateralAgreement(0.0, 0.0, independentAmount, 10.0);
  }

  public boolean isCollateralised() {
    return Double.isFinite(threshold);
  }

  /** Number of grid steps in the margin period of risk, at least one. */
  public int marginPeriodSteps(double stepYears) {
    if (!isCollateralised()) {
      return 0;
    }
    return Math.max(1, (int) Math.round(marginPeriodOfRiskDays / 365.0 / stepYears));
  }
}
