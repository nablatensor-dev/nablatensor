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
 * @param riskClass GIRR / CSR / equity / commodity / FX
 * @param measure   delta, vega or curvature
 * @param bucket    the risk-class bucket id (e.g. equity bucket {@code "5"})
 * @param name      the specific factor (issuer, curve tenor, index, ...)
 * @param tenor     option-expiry tenor in years for vega; {@code 0} otherwise
 */
public record RiskFactor(RiskClass riskClass, RiskMeasure measure, String bucket, String name, double tenor) {

  public RiskFactor(RiskClass riskClass, RiskMeasure measure, String bucket, String name) {
    this(riskClass, measure, bucket, name, 0.0);
  }

  public static RiskFactor equityDelta(String bucket, String name) {
    return new RiskFactor(RiskClass.EQUITY, RiskMeasure.DELTA, bucket, name);
  }

  public static RiskFactor equityVega(String bucket, String name, double tenor) {
    return new RiskFactor(RiskClass.EQUITY, RiskMeasure.VEGA, bucket, name, tenor);
  }

  public RiskFactor asCurvature() {
    return new RiskFactor(riskClass, RiskMeasure.CURVATURE, bucket, name, tenor);
  }
}
