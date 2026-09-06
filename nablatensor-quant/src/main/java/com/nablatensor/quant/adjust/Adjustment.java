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
package com.nablatensor.quant.adjust;

/**
 * A rate {@code adjustment} to be added to a base forward rate, and the
 * resulting {@code adjustedRate}. The sign convention is "add to the forward":
 * an in-arrears LIBOR fixing has a positive adjustment, a Eurodollar futures
 * convexity adjustment is quoted as the (positive) amount by which the futures
 * rate exceeds the forward.
 */
public record Adjustment(double baseRate, double adjustment) {

  public double adjustedRate() {
    return baseRate + adjustment;
  }

  /** Adjustment in basis points. */
  public double adjustmentBp() {
    return adjustment * 1.0e4;
  }
}
