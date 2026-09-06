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
 * The differentiable inputs a netting-set CVA is recorded against. Every
 * component is a {@code double}, so the gradient handed back by one adjoint
 * sweep is a {@code CvaMarket} of the same shape: {@link #r0()} / {@link #hwLevel()}
 * / {@link #hwMeanReversion()} / {@link #hwSigma()} carry the interest-rate
 * delta and rate-volatility sensitivity, {@link #hazardShort()} /
 * {@link #hazardMid()} / {@link #hazardLong()} carry the counterparty CS01 by
 * tenor bucket, {@link #recovery()} the recovery sensitivity, and the {@code fx*}
 * components the FX delta and FX-volatility sensitivity.
 *
 * @param r0              initial short rate (also the flat instantaneous forward)
 * @param hwLevel         Hull-White reversion level {@code b}
 * @param hwMeanReversion Hull-White reversion speed {@code a}
 * @param hwSigma         Hull-White absolute short-rate volatility
 * @param hazardShort     counterparty forward hazard on {@code [0, 2y]}
 * @param hazardMid       counterparty forward hazard on {@code [2y, 5y]}
 * @param hazardLong      counterparty forward hazard beyond {@code 5y}
 * @param recovery        counterparty recovery rate
 * @param fxSpot          reporting currency per unit foreign currency
 * @param fxVol           lognormal FX volatility
 * @param fxForeignRate   flat foreign-currency rate used for the FX forward
 */
public record CvaMarket(double r0, double hwLevel, double hwMeanReversion, double hwSigma,
                        double hazardShort, double hazardMid, double hazardLong, double recovery,
                        double fxSpot, double fxVol, double fxForeignRate) {

  public CvaMarket validated() {
    if (!(hwMeanReversion > 0.0) || !(hwSigma >= 0.0) || !(fxVol >= 0.0) || !(fxSpot > 0.0)) {
      throw new IllegalArgumentException("need hwMeanReversion>0, hwSigma>=0, fxVol>=0, fxSpot>0; got " + this);
    }
    if (!(recovery >= 0.0 && recovery < 1.0)) {
      throw new IllegalArgumentException("recovery must be in [0, 1), got " + recovery);
    }
    return this;
  }

  /** A textbook single-A counterparty world: 3% rates, 150 bp CDS at 40% recovery. */
  public static CvaMarket demo() {
    double lgd = 1.0 - 0.40;
    double lambda = 150.0e-4 / lgd;
    return new CvaMarket(0.03, 0.03, 0.10, 0.010,
        lambda, lambda, lambda, 0.40,
        1.10, 0.12, 0.024);
  }

  public CvaMarket withShortRate(double r0) {
    return new CvaMarket(r0, hwLevel, hwMeanReversion, hwSigma,
        hazardShort, hazardMid, hazardLong, recovery, fxSpot, fxVol, fxForeignRate);
  }

  public CvaMarket withHazardParallelShift(double delta) {
    return new CvaMarket(r0, hwLevel, hwMeanReversion, hwSigma,
        Math.max(0.0, hazardShort + delta), Math.max(0.0, hazardMid + delta),
        Math.max(0.0, hazardLong + delta), recovery, fxSpot, fxVol, fxForeignRate);
  }

  public CvaMarket withFxSpot(double fxSpot) {
    return new CvaMarket(r0, hwLevel, hwMeanReversion, hwSigma,
        hazardShort, hazardMid, hazardLong, recovery, fxSpot, fxVol, fxForeignRate);
  }

  public CvaMarket withCurveLevel(double r0, double hwLevel) {
    return new CvaMarket(r0, hwLevel, hwMeanReversion, hwSigma,
        hazardShort, hazardMid, hazardLong, recovery, fxSpot, fxVol, fxForeignRate);
  }

  public CvaMarket withRateVol(double hwSigma) {
    return new CvaMarket(r0, hwLevel, hwMeanReversion, hwSigma,
        hazardShort, hazardMid, hazardLong, recovery, fxSpot, fxVol, fxForeignRate);
  }

  public CvaMarket withHazards(double hazardShort, double hazardMid, double hazardLong) {
    return new CvaMarket(r0, hwLevel, hwMeanReversion, hwSigma,
        Math.max(0.0, hazardShort), Math.max(0.0, hazardMid), Math.max(0.0, hazardLong),
        recovery, fxSpot, fxVol, fxForeignRate);
  }

  public CvaMarket withFxVol(double fxVol) {
    return new CvaMarket(r0, hwLevel, hwMeanReversion, hwSigma,
        hazardShort, hazardMid, hazardLong, recovery, fxSpot, fxVol, fxForeignRate);
  }

  /**
   * Every component multiplied by {@code factor}. Used to lift a gradient that
   * was differentiated against a non-dimensionalised (money-scaled) CVA back
   * into reporting-currency units: {@code d(scale*CVA)/dx = scale * dCVA/dx}.
   */
  public CvaMarket scale(double factor) {
    return new CvaMarket(r0 * factor, hwLevel * factor, hwMeanReversion * factor, hwSigma * factor,
        hazardShort * factor, hazardMid * factor, hazardLong * factor, recovery * factor,
        fxSpot * factor, fxVol * factor, fxForeignRate * factor);
  }
}
