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
 * FRTB's three correlation scenarios (MAR21.6). The bank computes the capital
 * charge under each and takes the largest.
 */
public enum CorrelationScenario {
  /** Prescribed correlations as published. */
  MEDIUM,
  /** {@code min(1.25 * rho, 1)} — correlations move toward 1. */
  HIGH,
  /** {@code max(2 * rho - 1, 0.75 * rho)} — correlations move toward 0 (or negative). */
  LOW;

  /** Applies this scenario's transform to a medium-scenario correlation. */
  public double apply(double mediumRho) {
    return switch (this) {
      case MEDIUM -> mediumRho;
      case HIGH -> Math.min(1.25 * mediumRho, 1.0);
      case LOW -> Math.max(2.0 * mediumRho - 1.0, 0.75 * mediumRho);
    };
  }
}
