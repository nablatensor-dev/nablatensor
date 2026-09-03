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
 * The three PRA standardised CVA methods (Basel 3.1, PS1/26 — no internal
 * models), exposed over one {@link CvaCapital} run:
 *
 * <ul>
 *   <li><b>Alternative Approach</b> — the simplified method for smaller books:
 *       BA-CVA reduced, no hedge recognition, scaled by a supervisory factor.</li>
 *   <li><b>Basic Approach (BA-CVA)</b> — reduced or full with CDS hedge
 *       recognition.</li>
 *   <li><b>Standardised Approach (SA-CVA)</b> — the sensitivities-based charge.</li>
 * </ul>
 */
public record PraCvaMethods(CvaCapital capital, double alternativeApproachScalar) {

  /** The default {@code 1.0} scalar on the Alternative Approach (supervisor may set higher). */
  public static PraCvaMethods of(CvaCapital capital) {
    return new PraCvaMethods(capital, 1.0);
  }

  public double alternativeApproach() {
    return alternativeApproachScalar * capital.baCva().reduced();
  }

  public double basicApproach() {
    return capital.baCva().full();
  }

  public double standardisedApproach() {
    return capital.saCva().total();
  }

  /** The method a bank on all three would report — the binding (largest) charge. */
  public double bindingCharge() {
    return Math.max(alternativeApproach(), Math.max(basicApproach(), standardisedApproach()));
  }
}
