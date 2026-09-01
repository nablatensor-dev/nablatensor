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

/**
 * A guided tour of the FRTB standardised approach, as runnable tests. Read them
 * in order — each lesson is a small, self-contained explanation with a worked
 * example that the code reproduces. See {@code docs-reg/frtb-sa.md} for the
 * prose version.
 *
 * <pre>
 *   mechanics of the aggregation
 *     01  WeightedSensitivity          WS_k = RW_k * s_k
 *     02  WithinBucketCorrelation      K_b, and what rho does to it
 *     03  AcrossBucketAggregation      S_b, gamma, the second level
 *     04  AlternativeSpecificationFloor the max(0, .) that keeps charges real
 *     05  ThreeCorrelationScenarios    HIGH / MEDIUM / LOW, and which one binds
 *
 *   delta, one risk class at a time
 *     06  EquityDelta                  the first real MAR21 parameter table
 *     07  GirrTenorVertices            the yield-curve vertices and the decay
 *     08  GirrLiquidCurrencyRelief     the sqrt(2) divisor
 *     09  GirrInflationAndBasis        the two non-tenor GIRR factors
 *     10  CsrBondCdsBasis              rho_name * rho_tenor * rho_basis
 *     11  CommodityLocationBasis       the same idea, delivery-location edition
 *     12  FxRiskAndReportingCurrency   one factor per pair, gamma = 60%
 *
 *   vega and curvature
 *     13  VegaRisk                     the implied-vol charge
 *     14  CurvatureShock               two repricings, delta stripped out
 *     15  CurvaturePsiGating           why two long options do not diversify
 *
 *   default risk and residual risk
 *     16  JumpToDefault                gross JTD = LGD*notional + P&L
 *     17  DrcHedgeBenefitRatio         partial hedge recognition
 *     18  ResidualRiskAddOn            1.0% / 0.1% on gross notional
 *
 *   putting it together
 *     19  FullStandardisedApproach     SBM + DRC + RRAO
 *     20  RelievedUnrelievedDual       one book, two parameter sets
 * </pre>
 */
package com.nablatensor.reg.tour;
