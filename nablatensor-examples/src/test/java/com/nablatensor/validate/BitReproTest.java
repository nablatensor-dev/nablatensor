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
package com.nablatensor.validate;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.Products;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Every backend available on the build machine must reproduce the scalar CPU
 * oracle within tolerance, and the oracle's adjoint gradient must agree with a
 * central bump. This is the MVP "definition of done" turned into an assertion.
 */
@Tag("mc")
class BitReproTest {

  @Test
  void everyAvailableBackendReproducesTheOracle() {
    Report report = ModelValidation.of(Products.asianCall())
        .market(EquityMarket.atmOneYear())
        .steps(64)
        .scenarios(200_000L)
        .seed(20260830L)
        .fp64()
        .tolerance(1e-6)
        .run();

    System.out.println(report);
    assertTrue(report.passed(), report.firstFailure());
  }

  @Test
  void adjointDeltaAgreesWithCentralBump() {
    Report report = ModelValidation.of(Products.europeanCall())
        .market(EquityMarket.atmOneYear())
        .steps(1)
        .scenarios(200_000L)
        .seed(20260830L)
        .fp64()
        .bump(5e-3)
        .run();

    BumpCrossCheck x = report.bumpCrossCheck();
    assertTrue(x.absDiff().spot() < 5e-3,
        "adjoint delta vs bump: " + x.adjoint().spot() + " vs " + x.bump().spot());
  }
}
