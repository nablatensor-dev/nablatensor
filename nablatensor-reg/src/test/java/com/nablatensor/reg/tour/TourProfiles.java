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
package com.nablatensor.reg.tour;

import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskClassProfile;
import com.nablatensor.risk.RiskFactor;

/**
 * Toy {@link RiskClassProfile}s for the guided tour (lessons 1–5). They are not
 * a rulebook — they let a lesson dial one knob (a risk weight, a correlation) and
 * watch the aggregation react, before the real MAR21 tables arrive in lesson 6+.
 *
 * <p>All toy factors are keyed as {@code EQUITY} so a plain
 * {@code RiskFactor.equityDelta(bucket, name)} works as the key.
 */
final class TourProfiles {

  private TourProfiles() {
  }

  /**
   * A flat profile: one risk weight for every factor, one within-bucket
   * correlation for every distinct pair, one across-bucket correlation for every
   * distinct bucket pair. Curvature shock is {@code rw * level}.
   */
  static RiskClassProfile flat(double riskWeight, double withinBucketRho, double acrossBucketGamma) {
    return new RiskClassProfile() {
      @Override
      public RiskClass riskClass() {
        return RiskClass.EQUITY;
      }

      @Override
      public double deltaRiskWeight(RiskFactor k) {
        return riskWeight;
      }

      @Override
      public double vegaRiskWeight(RiskFactor k) {
        return riskWeight;
      }

      @Override
      public double curvatureShock(RiskFactor k, double riskFactorLevel) {
        return riskWeight * riskFactorLevel;
      }

      @Override
      public double deltaRho(RiskFactor k, RiskFactor l) {
        return k.equals(l) ? 1.0 : withinBucketRho;
      }

      @Override
      public double vegaRho(RiskFactor k, RiskFactor l) {
        return k.equals(l) ? 1.0 : withinBucketRho;
      }

      @Override
      public double gamma(String bucketB, String bucketC) {
        return bucketB.equals(bucketC) ? 1.0 : acrossBucketGamma;
      }
    };
  }
}
