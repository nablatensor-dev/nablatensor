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

import com.nablatensor.codegen.Of;

/**
 * A regulatory risk factor: the key a sensitivity is bucketed and weighted by.
 *
 * <p>The record is deliberately generic — one shape for every FRTB SA / SIMM
 * risk class. Interpretation of {@code name} / {@code tenor} / {@code tenor2} is
 * per risk class; the typed factory methods below encode the conventions the
 * FRTB SA / SIMM parameter classes expect:
 *
 * <ul>
 *   <li><b>GIRR</b> delta: {@code bucket} = currency, {@code name} = curve id
 *       ({@code "OIS"}, {@code "3M"}, {@code "INFL"}, {@code "XCCY"}),
 *       {@code tenor} = vertex in years.</li>
 *   <li><b>GIRR</b> vega: {@code tenor} = option maturity, {@code tenor2} =
 *       residual maturity of the underlying.</li>
 *   <li><b>CSR</b> delta: {@code bucket} = sector/quality bucket id,
 *       {@code name} = {@code "<issuer>|BOND"} or {@code "<issuer>|CDS"},
 *       {@code tenor} = vertex in years.</li>
 *   <li><b>Equity</b> delta: spot has {@code tenor == 0}; the repo-rate factor
 *       of the same issuer has {@code tenor > 0}.</li>
 *   <li><b>Commodity</b> delta: {@code name} = {@code "<commodity>|<location>"},
 *       {@code tenor} = maturity in years.</li>
 *   <li><b>FX</b> delta: {@code bucket} = {@code name} = the currency pair.</li>
 * </ul>
 *
 * @param riskClass GIRR / CSR / equity / commodity / FX
 * @param measure   delta, vega or curvature
 * @param bucket    the risk-class bucket id (numeric for most classes; a code for FX)
 * @param name      the specific factor (issuer, curve id, index, currency pair, ...)
 * @param tenor     primary tenor in years (curve vertex, option expiry, repo tenor); {@code 0} if not applicable
 * @param tenor2    secondary tenor in years (GIRR/CSR vega: underlying residual maturity); {@code 0} otherwise
 */
@Of
public final class RiskFactor {

  private RiskFactor(RiskClassEnum riskClass, RiskMeasureEnum measure, String bucket, String name, double tenor, double tenor2) {
    this.riskClass = riskClass;
    this.measure = measure;
    this.bucket = bucket;
    this.name = name;
    this.tenor = tenor;
    this.tenor2 = tenor2;
  }

  private final RiskClassEnum riskClass;
  private final RiskMeasureEnum measure;
  private final String bucket;
  private final String name;
  private final double tenor;
  private final double tenor2;
  static RiskFactor create(RiskClassEnum riskClass, RiskMeasureEnum measure, String bucket, String name, double tenor, double tenor2) {
    return new RiskFactor(riskClass, measure, bucket, name, tenor, tenor2);
  }

  public static RiskFactorBuilder of() {
    return new RiskFactorBuilder();
  }

  public RiskClassEnum riskClass() {
    return riskClass;
  }

  public RiskMeasureEnum measure() {
    return measure;
  }

  public String bucket() {
    return bucket;
  }

  public String name() {
    return name;
  }

  public double tenor() {
    return tenor;
  }

  public double tenor2() {
    return tenor2;
  }


  private RiskFactor(RiskClassEnum riskClass, RiskMeasureEnum measure, String bucket, String name) {
    this(riskClass, measure, bucket, name, 0.0, 0.0);
  }

  private RiskFactor(RiskClassEnum riskClass, RiskMeasureEnum measure, String bucket, String name, double tenor) {
    this(riskClass, measure, bucket, name, tenor, 0.0);
  }

  // ---- equity (spot) ----------------------------------------------------

  public static RiskFactor equityDelta(String bucket, String name) {
    return RiskFactor.of().riskClass(RiskClassEnum.EQUITY).measure(RiskMeasureEnum.DELTA).bucket(bucket).name(name).tenor(0.0).tenor2(0.0).build();
  }

  private static RiskFactor equityVega(String bucket, String name, double tenor) {
    return RiskFactor.of().riskClass(RiskClassEnum.EQUITY).measure(RiskMeasureEnum.VEGA).bucket(bucket).name(name).tenor(tenor).tenor2(0.0).build();
  }

  /** The equity repo-rate factor of an issuer (distinguished from spot by {@code tenor > 0}). */
  private static RiskFactor equityRepoDelta(String bucket, String issuer, double tenorYears) {
    if (tenorYears <= 0.0) {
      throw new IllegalArgumentException("equity repo tenor must be > 0 (spot uses equityDelta)");
    }
    return RiskFactor.of().riskClass(RiskClassEnum.EQUITY).measure(RiskMeasureEnum.DELTA).bucket(bucket).name(issuer).tenor(tenorYears).tenor2(0.0).build();
  }

  /** True for an equity repo-rate delta factor (as opposed to spot). */
  public boolean isEquityRepo() {
    return riskClass == RiskClassEnum.EQUITY && measure == RiskMeasureEnum.DELTA && tenor > 0.0;
  }

  // ---- GIRR -----------------------------------------------------------

  private static RiskFactor girrDelta(String ccy, String curveId, double vertexYears) {
    return RiskFactor.of().riskClass(RiskClassEnum.GIRR).measure(RiskMeasureEnum.DELTA).bucket(ccy).name(curveId).tenor(vertexYears).tenor2(0.0).build();
  }

  public static RiskFactor girrDelta(String ccy, double vertexYears) {
    return girrDelta(ccy, "OIS", vertexYears);
  }

  public static RiskFactor girrInflation(String ccy) {
    return RiskFactor.of().riskClass(RiskClassEnum.GIRR).measure(RiskMeasureEnum.DELTA).bucket(ccy).name("INFL").tenor(0.0).tenor2(0.0).build();
  }

  public static RiskFactor girrXccyBasis(String ccy) {
    return RiskFactor.of().riskClass(RiskClassEnum.GIRR).measure(RiskMeasureEnum.DELTA).bucket(ccy).name("XCCY").tenor(0.0).tenor2(0.0).build();
  }

  private static RiskFactor girrVega(String ccy, double optionMaturityYears, double underlyingMaturityYears) {
    return RiskFactor.of().riskClass(RiskClassEnum.GIRR).measure(RiskMeasureEnum.VEGA).bucket(ccy).name("VOL").tenor(optionMaturityYears).tenor2(underlyingMaturityYears).build();
  }

  // ---- CSR (non-securitisation; the same shape serves sec / CTP) -------

  private static RiskFactor csrDelta(String bucket, String issuer, CsrCurveEnum curve, double vertexYears) {
    return RiskFactor.of().riskClass(RiskClassEnum.CSR_NON_SEC).measure(RiskMeasureEnum.DELTA).bucket(bucket).name(issuer + "|" + curve.name()).tenor(vertexYears).tenor2(0.0).build();
  }

  private static RiskFactor csrDelta(RiskClassEnum csrClass, String bucket, String issuer, CsrCurveEnum curve, double vertexYears) {
    return RiskFactor.of().riskClass(csrClass).measure(RiskMeasureEnum.DELTA).bucket(bucket).name(issuer + "|" + curve.name()).tenor(vertexYears).tenor2(0.0).build();
  }

  private static RiskFactor csrVega(String bucket, String issuer, double optionMaturityYears) {
    return RiskFactor.of().riskClass(RiskClassEnum.CSR_NON_SEC).measure(RiskMeasureEnum.VEGA).bucket(bucket).name(issuer + "|VOL").tenor(optionMaturityYears).tenor2(0.0).build();
  }

  /** Bond vs CDS credit-spread curve — the CSR "basis" pair. */
  public enum CsrCurveEnum {
    /** Spread implied by the issuer's cash bonds. */
    BOND,
    /** Spread quoted in the issuer's credit default swaps. */
    CDS
  }

  /** The issuer part of a CSR factor name ({@code "<issuer>|BOND"} -> {@code "<issuer>"}). */
  public String csrIssuer() {
    int bar = name.indexOf('|');
    return bar < 0 ? name : name.substring(0, bar);
  }

  /** The curve part of a CSR factor name ({@code "BOND"} / {@code "CDS"} / {@code "VOL"}); {@code ""} if none. */
  public String csrCurve() {
    int bar = name.indexOf('|');
    return bar < 0 ? "" : name.substring(bar + 1);
  }

  // ---- commodity ----------------------------------------------------

  private static RiskFactor commodityDelta(String bucket, String commodity, double maturityYears, String deliveryLocation) {
    return RiskFactor.of().riskClass(RiskClassEnum.COMMODITY).measure(RiskMeasureEnum.DELTA).bucket(bucket).name(commodity + "|" + deliveryLocation).tenor(maturityYears).tenor2(0.0).build();
  }

  private static RiskFactor commodityVega(String bucket, String commodity, double optionMaturityYears) {
    return RiskFactor.of().riskClass(RiskClassEnum.COMMODITY).measure(RiskMeasureEnum.VEGA).bucket(bucket).name(commodity + "|VOL").tenor(optionMaturityYears).tenor2(0.0).build();
  }

  /** The commodity part of a commodity factor name ({@code "WTI|HUB"} -> {@code "WTI"}). */
  public String commodityName() {
    int bar = name.indexOf('|');
    return bar < 0 ? name : name.substring(0, bar);
  }

  /** The delivery-location part of a commodity factor name; {@code ""} if none. */
  public String deliveryLocation() {
    int bar = name.indexOf('|');
    return bar < 0 ? "" : name.substring(bar + 1);
  }

  // ---- FX -----------------------------------------------------------

  public static RiskFactor fxDelta(String currencyPair) {
    return RiskFactor.of().riskClass(RiskClassEnum.FX).measure(RiskMeasureEnum.DELTA).bucket(currencyPair).name(currencyPair).tenor(0.0).tenor2(0.0).build();
  }

  public static RiskFactor fxVega(String currencyPair, double optionMaturityYears) {
    return RiskFactor.of().riskClass(RiskClassEnum.FX).measure(RiskMeasureEnum.VEGA).bucket(currencyPair).name(currencyPair).tenor(optionMaturityYears).tenor2(0.0).build();
  }

  // ---- curvature --------------------------------------------------

  /** The same factor as a curvature factor (measure {@code CURVATURE}, tenors preserved). */
  public RiskFactor asCurvature() {
    return RiskFactor.of().riskClass(riskClass).measure(RiskMeasureEnum.CURVATURE).bucket(bucket).name(name).tenor(tenor).tenor2(tenor2).build();
  }

  /** A curvature factor with the tenor collapsed — one curvature factor per curve (GIRR / CSR). */
  public RiskFactor asCurvatureCurve() {
    return RiskFactor.of().riskClass(riskClass).measure(RiskMeasureEnum.CURVATURE).bucket(bucket).name(name).tenor(0.0).tenor2(0.0).build();
  }
}
