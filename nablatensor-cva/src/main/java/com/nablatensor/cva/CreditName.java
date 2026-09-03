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
 * A credit name — the counterparty of a netting set or the reference of a CDS
 * hedge — with the curve that drives its default and the rating / sector keys
 * the BA-CVA and SA-CVA parameter tables look up.
 *
 * @param id       identifier, unique within a portfolio
 * @param curve    the counterparty hazard curve
 * @param recovery assumed recovery rate on default, in {@code [0, 1)}
 * @param rating   credit-quality bucket for the BA-CVA risk weight
 * @param sector   sector bucket for the BA-CVA risk weight and SA-CVA correlations
 */
public record CreditName(String id, HazardCurve curve, double recovery,
                         Rating rating, Sector sector) {

  public CreditName {
    if (!(recovery >= 0.0 && recovery < 1.0)) {
      throw new IllegalArgumentException("recovery must be in [0, 1), got " + recovery);
    }
  }

  public double lossGivenDefault() {
    return 1.0 - recovery;
  }

  /** Credit-quality buckets used by the BA-CVA risk-weight table (MAR50.5). */
  public enum Rating { AAA, AA, A, BBB, BB, B, CCC, UNRATED }

  /** Sector buckets used by the BA-CVA risk weights and the SA-CVA correlations. */
  public enum Sector { SOVEREIGN, LOCAL_GOVERNMENT, FINANCIAL, CORPORATE, CONSUMER, TECH, OTHER }
}
