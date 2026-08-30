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
package com.nablatensor.quant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.engine.SDouble;
import com.nablatensor.engine.Nabla;
import com.nablatensor.ops.Smooth;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * One recorded tape, one compiled kernel, N named risk measures: the selector-weight
 * trick must return each measure's value and its full input gradient, matching a
 * standalone single-output run for that measure.
 */
class MultiOutputTest {

  private static final long N = 400_000L;
  private static final long SEED = 4242L;
  private static final int STEPS = 20;

  /** Terminal-spot GBM with a call and a smoothed digital, sharing the path. */
  private static Map<String, SDouble> measures(com.nablatensor.engine.AadRecorder rec) {
    SDouble s0 = rec.input("S0", 100.0);
    SDouble k = rec.input("K", 100.0);
    SDouble vol = rec.input("sigma", 0.2);
    SDouble r = rec.input("r", 0.03);
    double dt = 1.0 / STEPS;
    SDouble drift = r.sub(vol.mul(vol).mul(0.5)).mul(dt);
    SDouble vs = vol.mul(Math.sqrt(dt));
    SDouble s = s0;
    for (int t = 0; t < STEPS; t++) {
      s = s.mul(drift.add(vs.mul(rec.randn())).exp());
    }
    SDouble disc = r.neg().exp();
    Map<String, SDouble> m = new LinkedHashMap<>();
    m.put("call", s.sub(k).max(0.0).mul(disc));
    m.put("digital", Smooth.gt(rec, s, k, 1.0).mul(disc));
    return m;
  }

  @Test
  void valuesAndGradientsMatchStandaloneRuns() {
    try (MultiOutput mo = MultiOutput.of(MultiOutputTest::measures).on("cpu-jit").build()) {
      MultiOutput.Result r = mo.run(N, SEED);

      // --- standalone single-output kernels for the same two measures, same seed ---
      double callPx;
      double callVega;
      try (Nabla.Pricer p = Nabla.model(rec -> {
            Map<String, SDouble> m = measures(rec);
            rec.output(m.get("call"));
          }).fp64().greeks().on("cpu-jit").build()) {
        Nabla.Valuation v = p.value().scenarios(N).seed(SEED).run();
        callPx = v.price();
        callVega = v.greek("sigma");
      }
      double digitalPx;
      try (Nabla.Pricer p = Nabla.model(rec -> {
            Map<String, SDouble> m = measures(rec);
            rec.output(m.get("digital"));
          }).fp64().greeks().on("cpu-jit").build()) {
        digitalPx = p.value().scenarios(N).seed(SEED).run().price();
      }

      assertEquals(callPx, r.value("call"), 1e-9 * (1 + callPx), "call value: multi-output vs standalone");
      assertEquals(digitalPx, r.value("digital"), 1e-9 * (1 + digitalPx), "digital value");
      assertEquals(callVega, r.sensitivity("call", "sigma"), 1e-9 * (1 + Math.abs(callVega)), "call vega");

      // digital delta should be positive and finite
      assertTrue(r.sensitivity("digital", "S0") > 0.0, "digital delta > 0");
      assertEquals(2, r.outputOrder().size());
    }
  }

  @Test
  void marketOverridesFeedThroughEveryPass() {
    try (MultiOutput mo = MultiOutput.of(MultiOutputTest::measures).on("cpu-jit").build()) {
      double base = mo.run(N, SEED).value("call");
      double bumped = mo.run(Map.of("S0", 105.0), N, SEED).value("call");
      assertTrue(bumped > base + 1.0, "call value rises when S0 is overridden upward");
    }
  }
}
