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

/**
 * FRTB SA <b>Default Risk Charge</b> parameters, from the Basel Framework MAR22:
 * loss-given-default by seniority (MAR22.14) and the default risk weight by
 * credit-quality category (MAR22.24).
 *
 * <p>This is a data table; check it against your regulator's current rulebook.
 */
public final class DrcParameters {

  private DrcParameters() {
  }

  /** Seniority of a debt/equity position, driving loss-given-default. */
  public enum Seniority {
    SENIOR(0.25),
    NON_SENIOR(0.75),
    COVERED_BOND(0.25),
    EQUITY(1.00);

    private final double lgd;

    Seniority(double lgd) {
      this.lgd = lgd;
    }

    /** Loss-given-default as a decimal fraction. */
    public double lgd() {
      return lgd;
    }
  }

  /** Credit-quality category of the obligor, driving the default risk weight. */
  public enum CreditQuality {
    AAA(0.005),
    AA(0.02),
    A(0.03),
    BBB(0.06),
    BB(0.15),
    B(0.30),
    CCC(0.50),
    UNRATED(0.15),
    DEFAULTED(1.00);

    private final double riskWeight;

    CreditQuality(double riskWeight) {
      this.riskWeight = riskWeight;
    }

    /** Default risk weight as a decimal fraction. */
    public double riskWeight() {
      return riskWeight;
    }
  }

  /** The three DRC non-securitisation buckets (MAR22.23); no diversification across them. */
  public enum DrcBucket {
    CORPORATES,
    SOVEREIGNS,
    LOCAL_GOVERNMENT
  }
}
