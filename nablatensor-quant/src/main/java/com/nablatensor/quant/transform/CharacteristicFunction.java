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
package com.nablatensor.quant.transform;

/**
 * The risk-neutral characteristic function of the log-return
 * {@code X_T = ln(S_T / S_0)}: {@code phi(u, T) = E^Q[ exp(i u X_T) ]}. The drift
 * and any martingale correction are baked in, so {@link CosMethod} needs nothing
 * else to price a European option.
 *
 * <p>Implementations also supply the first, second and fourth cumulants of
 * {@code X_T}, which set the COS truncation range.
 */
public interface CharacteristicFunction {

  Complex phi(double u, double maturity);

  /** First cumulant (mean) of {@code X_T}. */
  double cumulant1(double maturity);

  /** Second cumulant (variance) of {@code X_T}. */
  double cumulant2(double maturity);

  /** Fourth cumulant of {@code X_T} ({@code 0} if unknown; the range then leans on {@link #cumulant2}). */
  default double cumulant4(double maturity) {
    return 0.0;
  }
}
