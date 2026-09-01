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
package com.nablatensor.reg.frtb.sbm;

import com.nablatensor.risk.RiskFactor;

/**
 * The two shocked repricings MAR21.5 needs for one curvature risk factor, plus
 * the inputs to strip the linear (delta) part out of them.
 *
 * <p>The engine forms
 * <pre>{@code
 * shock = profile.curvatureShock(factor, riskFactorLevel)
 * up   = pvUp   - pvBase - shock * netDelta
 * down = pvDown - pvBase + shock * netDelta
 * CVR  = -min(up, down)
 * }</pre>
 * so the caller supplies the base PV and the PV after moving the risk factor up
 * / down by the prescribed shock, together with the book's net delta to that
 * factor. For curve classes (GIRR / CSR) provide <em>one</em>
 * {@code CurvatureRepricing} per curve, keyed by a curve-level
 * {@link RiskFactor} (use {@link RiskFactor#asCurvatureCurve()} shape).
 *
 * @param factor          the curvature risk factor (any measure; the engine re-keys it)
 * @param riskFactorLevel the current level of the risk factor (spot, rate, fx, ...)
 * @param netDelta        the book's net delta to this factor (dPV/d(level))
 * @param pvBase          book PV at the base market
 * @param pvUp            book PV after the up shock
 * @param pvDown          book PV after the down shock
 */
public record CurvatureRepricing(RiskFactor factor, double riskFactorLevel, double netDelta,
                                 double pvBase, double pvUp, double pvDown) {
}
