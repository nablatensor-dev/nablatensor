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

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.Nabla;
import com.nablatensor.engine.SDouble;
import com.nablatensor.ops.Smooth;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Native multi-output: one tape with several {@code rec.output(name, ...)} calls,
 * one compiled kernel, one replay running the forward sweep once and one reverse
 * sweep per output. Each output's value and full gradient must match a standalone
 * single-output kernel recorded from the same body at the same seed
 * <em>bit-for-bit</em> — the forward sweep, and therefore every shared node, is
 * literally the same computation.
 */
@Tag("mc")
class MultiOutputNativeTest {

  private static final long N = 300_000L;
  private static final long SEED = 20260901L;
  private static final int STEPS = 24;

  private static Map<String, SDouble> measures(AadRecorder rec) {
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
    m.put("straddle", s.sub(k).abs().mul(disc));
    m.put("digital", Smooth.gt(rec, s, k, 1.0).mul(disc));
    return m;
  }

  @Test
  void eachOutputIsBitIdenticalToItsStandaloneKernel() {
    for (String engine : new String[] {"cpu", "cpu-jit"}) {
      try (MultiOutput mo = MultiOutput.of(MultiOutputNativeTest::measures).on(engine).build()) {
        MultiOutput.Result r = mo.run(N, SEED);
        assertEquals(3, r.outputOrder().size());

        for (String name : r.outputOrder()) {
          double v;
          double vega;
          double delta;
          try (Nabla.Pricer p = Nabla.model(rec -> {
                Map<String, SDouble> m = measures(rec);
                rec.output(m.get(name));
              }).fp64().greeks().on(engine).build()) {
            Nabla.Valuation val = p.value().scenarios(N).seed(SEED).run();
            v = val.price();
            vega = val.greek("sigma");
            delta = val.greek("S0");
          }
          assertEquals(v, r.value(name), 0.0, engine + " " + name + " value bit-identical");
          assertEquals(vega, r.sensitivity(name, "sigma"), 0.0, engine + " " + name + " vega bit-identical");
          assertEquals(delta, r.sensitivity(name, "S0"), 0.0, engine + " " + name + " delta bit-identical");
          assertTrue(r.standardError(name) > 0.0 && Double.isFinite(r.standardError(name)),
              engine + " " + name + " has a finite standard error");
        }
      }
    }
  }
}
