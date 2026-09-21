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

import com.nablatensor.codegen.Of;

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
@Of
public final class CvaMarket {

  private CvaMarket(double r0, double hwLevel, double hwMeanReversion, double hwSigma, double hazardShort, double hazardMid, double hazardLong, double recovery, double fxSpot, double fxVol, double fxForeignRate) {
    this.r0 = r0;
    this.hwLevel = hwLevel;
    this.hwMeanReversion = hwMeanReversion;
    this.hwSigma = hwSigma;
    this.hazardShort = hazardShort;
    this.hazardMid = hazardMid;
    this.hazardLong = hazardLong;
    this.recovery = recovery;
    this.fxSpot = fxSpot;
    this.fxVol = fxVol;
    this.fxForeignRate = fxForeignRate;
  }

  private final double r0;
  private final double hwLevel;
  private final double hwMeanReversion;
  private final double hwSigma;
  private final double hazardShort;
  private final double hazardMid;
  private final double hazardLong;
  private final double recovery;
  private final double fxSpot;
  private final double fxVol;
  private final double fxForeignRate;
  static CvaMarket create(double r0, double hwLevel, double hwMeanReversion, double hwSigma, double hazardShort, double hazardMid, double hazardLong, double recovery, double fxSpot, double fxVol, double fxForeignRate) {
    return new CvaMarket(r0, hwLevel, hwMeanReversion, hwSigma, hazardShort, hazardMid, hazardLong, recovery, fxSpot, fxVol, fxForeignRate);
  }

  public static CvaMarketBuilder of() {
    return new CvaMarketBuilder();
  }

  public double r0() {
    return r0;
  }

  public double hwLevel() {
    return hwLevel;
  }

  public double hwMeanReversion() {
    return hwMeanReversion;
  }

  public double hwSigma() {
    return hwSigma;
  }

  public double hazardShort() {
    return hazardShort;
  }

  public double hazardMid() {
    return hazardMid;
  }

  public double hazardLong() {
    return hazardLong;
  }

  public double recovery() {
    return recovery;
  }

  public double fxSpot() {
    return fxSpot;
  }

  public double fxVol() {
    return fxVol;
  }

  public double fxForeignRate() {
    return fxForeignRate;
  }


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
    return CvaMarket.of().r0(0.03).hwLevel(0.03).hwMeanReversion(0.10).hwSigma(0.010).hazardShort(lambda).hazardMid(lambda).hazardLong(lambda).recovery(0.40).fxSpot(1.10).fxVol(0.12).fxForeignRate(0.024).build();
  }

  public CvaMarket withShortRate(double r0) {
    return CvaMarket.of().r0(r0).hwLevel(hwLevel).hwMeanReversion(hwMeanReversion).hwSigma(hwSigma).hazardShort(hazardShort).hazardMid(hazardMid).hazardLong(hazardLong).recovery(recovery).fxSpot(fxSpot).fxVol(fxVol).fxForeignRate(fxForeignRate).build();
  }

  public CvaMarket withHazardParallelShift(double delta) {
    return CvaMarket.of().r0(r0).hwLevel(hwLevel).hwMeanReversion(hwMeanReversion).hwSigma(hwSigma).hazardShort(Math.max(0.0, hazardShort + delta)).hazardMid(Math.max(0.0, hazardMid + delta)).hazardLong(Math.max(0.0, hazardLong + delta)).recovery(recovery).fxSpot(fxSpot).fxVol(fxVol).fxForeignRate(fxForeignRate).build();
  }

  public CvaMarket withFxSpot(double fxSpot) {
    return CvaMarket.of().r0(r0).hwLevel(hwLevel).hwMeanReversion(hwMeanReversion).hwSigma(hwSigma).hazardShort(hazardShort).hazardMid(hazardMid).hazardLong(hazardLong).recovery(recovery).fxSpot(fxSpot).fxVol(fxVol).fxForeignRate(fxForeignRate).build();
  }

  public CvaMarket withCurveLevel(double r0, double hwLevel) {
    return CvaMarket.of().r0(r0).hwLevel(hwLevel).hwMeanReversion(hwMeanReversion).hwSigma(hwSigma).hazardShort(hazardShort).hazardMid(hazardMid).hazardLong(hazardLong).recovery(recovery).fxSpot(fxSpot).fxVol(fxVol).fxForeignRate(fxForeignRate).build();
  }

  public CvaMarket withRateVol(double hwSigma) {
    return CvaMarket.of().r0(r0).hwLevel(hwLevel).hwMeanReversion(hwMeanReversion).hwSigma(hwSigma).hazardShort(hazardShort).hazardMid(hazardMid).hazardLong(hazardLong).recovery(recovery).fxSpot(fxSpot).fxVol(fxVol).fxForeignRate(fxForeignRate).build();
  }

  public CvaMarket withHazards(double hazardShort, double hazardMid, double hazardLong) {
    return CvaMarket.of().r0(r0).hwLevel(hwLevel).hwMeanReversion(hwMeanReversion).hwSigma(hwSigma).hazardShort(Math.max(0.0, hazardShort)).hazardMid(Math.max(0.0, hazardMid)).hazardLong(Math.max(0.0, hazardLong)).recovery(recovery).fxSpot(fxSpot).fxVol(fxVol).fxForeignRate(fxForeignRate).build();
  }

  public CvaMarket withFxVol(double fxVol) {
    return CvaMarket.of().r0(r0).hwLevel(hwLevel).hwMeanReversion(hwMeanReversion).hwSigma(hwSigma).hazardShort(hazardShort).hazardMid(hazardMid).hazardLong(hazardLong).recovery(recovery).fxSpot(fxSpot).fxVol(fxVol).fxForeignRate(fxForeignRate).build();
  }

  /**
   * Every component multiplied by {@code factor}. Used to lift a gradient that
   * was differentiated against a non-dimensionalised (money-scaled) CVA back
   * into reporting-currency units: {@code d(scale*CVA)/dx = scale * dCVA/dx}.
   */
  public CvaMarket scale(double factor) {
    return CvaMarket.of().r0(r0 * factor).hwLevel(hwLevel * factor).hwMeanReversion(hwMeanReversion * factor).hwSigma(hwSigma * factor).hazardShort(hazardShort * factor).hazardMid(hazardMid * factor).hazardLong(hazardLong * factor).recovery(recovery * factor).fxSpot(fxSpot * factor).fxVol(fxVol * factor).fxForeignRate(fxForeignRate * factor).build();
  }
}
