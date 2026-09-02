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
public record RiskFactor(RiskClass riskClass, RiskMeasure measure, String bucket, String name,
                         double tenor, double tenor2) {

  public RiskFactor(RiskClass riskClass, RiskMeasure measure, String bucket, String name) {
    this(riskClass, measure, bucket, name, 0.0, 0.0);
  }

  public RiskFactor(RiskClass riskClass, RiskMeasure measure, String bucket, String name, double tenor) {
    this(riskClass, measure, bucket, name, tenor, 0.0);
  }

  // ---- equity (spot) ----------------------------------------------------

  public static RiskFactor equityDelta(String bucket, String name) {
    return new RiskFactor(RiskClass.EQUITY, RiskMeasure.DELTA, bucket, name);
  }

  public static RiskFactor equityVega(String bucket, String name, double tenor) {
    return new RiskFactor(RiskClass.EQUITY, RiskMeasure.VEGA, bucket, name, tenor);
  }

  /** The equity repo-rate factor of an issuer (distinguished from spot by {@code tenor > 0}). */
  public static RiskFactor equityRepoDelta(String bucket, String issuer, double tenorYears) {
    if (tenorYears <= 0.0) {
      throw new IllegalArgumentException("equity repo tenor must be > 0 (spot uses equityDelta)");
    }
    return new RiskFactor(RiskClass.EQUITY, RiskMeasure.DELTA, bucket, issuer, tenorYears);
  }

  /** True for an equity repo-rate delta factor (as opposed to spot). */
  public boolean isEquityRepo() {
    return riskClass == RiskClass.EQUITY && measure == RiskMeasure.DELTA && tenor > 0.0;
  }

  // ---- GIRR -----------------------------------------------------------

  public static RiskFactor girrDelta(String ccy, String curveId, double vertexYears) {
    return new RiskFactor(RiskClass.GIRR, RiskMeasure.DELTA, ccy, curveId, vertexYears);
  }

  public static RiskFactor girrDelta(String ccy, double vertexYears) {
    return girrDelta(ccy, "OIS", vertexYears);
  }

  public static RiskFactor girrInflation(String ccy) {
    return new RiskFactor(RiskClass.GIRR, RiskMeasure.DELTA, ccy, "INFL", 0.0);
  }

  public static RiskFactor girrXccyBasis(String ccy) {
    return new RiskFactor(RiskClass.GIRR, RiskMeasure.DELTA, ccy, "XCCY", 0.0);
  }

  public static RiskFactor girrVega(String ccy, double optionMaturityYears, double underlyingMaturityYears) {
    return new RiskFactor(RiskClass.GIRR, RiskMeasure.VEGA, ccy, "VOL",
        optionMaturityYears, underlyingMaturityYears);
  }

  // ---- CSR (non-securitisation; the same shape serves sec / CTP) -------

  public static RiskFactor csrDelta(String bucket, String issuer, CsrCurve curve, double vertexYears) {
    return new RiskFactor(RiskClass.CSR_NON_SEC, RiskMeasure.DELTA, bucket,
        issuer + "|" + curve.name(), vertexYears);
  }

  public static RiskFactor csrDelta(RiskClass csrClass, String bucket, String issuer, CsrCurve curve, double vertexYears) {
    return new RiskFactor(csrClass, RiskMeasure.DELTA, bucket, issuer + "|" + curve.name(), vertexYears);
  }

  public static RiskFactor csrVega(String bucket, String issuer, double optionMaturityYears) {
    return new RiskFactor(RiskClass.CSR_NON_SEC, RiskMeasure.VEGA, bucket, issuer + "|VOL", optionMaturityYears);
  }

  /** Bond vs CDS credit-spread curve — the CSR "basis" pair. */
  public enum CsrCurve { BOND, CDS }

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

  public static RiskFactor commodityDelta(String bucket, String commodity, double maturityYears, String deliveryLocation) {
    return new RiskFactor(RiskClass.COMMODITY, RiskMeasure.DELTA, bucket,
        commodity + "|" + deliveryLocation, maturityYears);
  }

  public static RiskFactor commodityVega(String bucket, String commodity, double optionMaturityYears) {
    return new RiskFactor(RiskClass.COMMODITY, RiskMeasure.VEGA, bucket, commodity + "|VOL", optionMaturityYears);
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
    return new RiskFactor(RiskClass.FX, RiskMeasure.DELTA, currencyPair, currencyPair);
  }

  public static RiskFactor fxVega(String currencyPair, double optionMaturityYears) {
    return new RiskFactor(RiskClass.FX, RiskMeasure.VEGA, currencyPair, currencyPair, optionMaturityYears);
  }

  // ---- curvature --------------------------------------------------

  /** The same factor as a curvature factor (measure {@code CURVATURE}, tenors preserved). */
  public RiskFactor asCurvature() {
    return new RiskFactor(riskClass, RiskMeasure.CURVATURE, bucket, name, tenor, tenor2);
  }

  /** A curvature factor with the tenor collapsed — one curvature factor per curve (GIRR / CSR). */
  public RiskFactor asCurvatureCurve() {
    return new RiskFactor(riskClass, RiskMeasure.CURVATURE, bucket, name, 0.0, 0.0);
  }
}
