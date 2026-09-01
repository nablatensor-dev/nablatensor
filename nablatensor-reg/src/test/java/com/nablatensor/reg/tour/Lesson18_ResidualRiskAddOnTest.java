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

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.nablatensor.reg.frtb.rrao.Rrao;
import com.nablatensor.reg.frtb.rrao.Rrao.ResidualRiskKind;
import com.nablatensor.reg.frtb.rrao.Rrao.ResidualRiskPosition;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * LESSON 18 — the Residual Risk Add-On (MAR23): a deliberately crude backstop
 * for risks the SBM + DRC framework cannot see. No sensitivities, no netting —
 * just a weight on <b>gross notional</b>:
 *
 * <pre>
 *   RRAO = 1.0% * SUM |notional| for instruments with an EXOTIC underlying
 *        + 0.1% * SUM |notional| for instruments with OTHER residual risk
 *                                (gap / correlation / behavioural / dividend ...)
 * </pre>
 *
 * Because there is no netting, two offsetting digital options both attract the
 * add-on.
 */
class Lesson18_ResidualRiskAddOnTest {

  @Test
  void oneAndTenthPercentOnGrossNotional() {
    List<ResidualRiskPosition> book = List.of(
        new ResidualRiskPosition("longevity-swap", 2_000_000, ResidualRiskKind.EXOTIC_UNDERLYING),
        new ResidualRiskPosition("digital-call", 1_000_000, ResidualRiskKind.GAP_RISK));

    Rrao.Result r = Rrao.of(book).compute();
    assertEquals(0.010 * 2_000_000, r.exotic(), 1e-9);
    assertEquals(0.001 * 1_000_000, r.other(), 1e-9);
    assertEquals(20_000 + 1_000, r.total(), 1e-9);
  }

  @Test
  void thereIsNoNetting_offsettingDigitalsBothCharge() {
    List<ResidualRiskPosition> book = List.of(
        new ResidualRiskPosition("digital-long", 1_000_000, ResidualRiskKind.GAP_RISK),
        new ResidualRiskPosition("digital-short", -1_000_000, ResidualRiskKind.GAP_RISK));

    // |+1m| * 0.1%  +  |-1m| * 0.1%  = 1000 + 1000
    assertEquals(2_000.0, Rrao.of(book).compute().total(), 1e-9);
  }
}
