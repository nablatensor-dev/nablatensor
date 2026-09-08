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
package com.nablatensor.scenario;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Products;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** The scenario DSL: shocks apply correctly, ladders/grids expand, and a
 *  ladder re-price recovers the adjoint delta at the base point. */
class ScenarioTest {

  private static final EquityMarket M = EquityMarket.atmOneYear();

  @Test
  void shocksApplyByKind() {
    Map<String, Double> base = Map.of("spot", 100.0, "vol", 0.2);
    assertEquals(110.0, Shock.absolute("spot", 110).shocked(100.0), 0);
    assertEquals(105.0, Shock.relative("spot", 0.05).shocked(100.0), 0);
    assertEquals(0.25, Shock.additive("vol", 0.05).shocked(0.2), 1e-12);

    Map<String, Double> shocked = Scenario.of("s",
        Shock.relative("spot", 0.1), Shock.additive("vol", -0.03)).apply(base);
    assertEquals(110.0, shocked.get("spot"), 1e-9);
    assertEquals(0.17, shocked.get("vol"), 1e-9);
  }

  @Test
  void ladderAndGridExpansion() {
    Ladder spot = Ladder.of("spot").absolute().from(80).to(120).step(10);
    assertEquals(5, spot.size());
    assertEquals(5, ScenarioSet.ladder(spot).size());

    Ladder vol = Ladder.of("vol").additive().from(-0.05).to(0.05).points(3);
    assertEquals(15, ScenarioSet.grid(spot, vol).size());
    // each grid scenario carries one shock from each ladder
    ScenarioSet g = ScenarioSet.grid(spot, vol);
    assertTrue(g.scenarios().stream().allMatch(s -> s.shocks().size() == 2));
  }

  @Test
  void spotLadderIsMonotoneAndRecoversTheAdjointDelta() {
    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.europeanCall()).market(M).steps(1)
        .fp64().greeks().on("cpu-jit").build()) {

      Ladder ladder = Ladder.of("spot").absolute().from(90).to(110).step(2);
      ScenarioRunner.LadderResult r = ScenarioRunner.ladder(mc, M, ladder, 400_000L, 7L);

      for (int i = 1; i < r.price().length; i++) {
        assertTrue(r.price()[i] >= r.price()[i - 1] - 1e-9, "call price monotone in spot along the ladder");
      }
      // central difference of the ladder around spot=100 vs the adjoint delta there
      int mid = java.util.Arrays.binarySearch(r.x(), 100.0);
      double fd = (r.price()[mid + 1] - r.price()[mid - 1]) / (r.x()[mid + 1] - r.x()[mid - 1]);
      assertEquals(r.delta()[mid], fd, 5e-3 * (1 + fd), "ladder finite-difference vs adjoint delta at ATM");
    }
  }

  @Test
  void runnerAppliesEveryScenarioToABuiltKernel() {
    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.europeanCall()).market(M).steps(1)
        .fp64().priceOnly().on("cpu-jit").build()) {
      var set = ScenarioSet.list(
          Scenario.of("base"),
          Scenario.of("crash", Shock.relative("spot", -0.30)),
          Scenario.of("rally", Shock.relative("spot", 0.30)));
      var out = ScenarioRunner.run(mc, M, set, 300_000L, 7L);
      assertTrue(out.get("crash").price() < out.get("base").price());
      assertTrue(out.get("rally").price() > out.get("base").price());
    }
  }
}
