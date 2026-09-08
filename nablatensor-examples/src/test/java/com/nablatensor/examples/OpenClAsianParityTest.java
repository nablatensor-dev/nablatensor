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
package com.nablatensor.examples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadEngines;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.SDouble;
import java.util.Locale;
import org.junit.jupiter.api.Test;

/**
 * Exercises the OpenCL replay engine on a textbook arithmetic-average Asian call
 * Monte Carlo: one recorded tape, priced with pathwise-adjoint delta and vega,
 * replayed on OpenCL and diffed against the {@code cpu-jit} oracle, then hammered
 * to check the device path is stable — every repeat of a fixed
 * {@code (paths, seed)} must return bit-identical numbers and none may throw.
 *
 * <p>Skips unless an OpenCL engine is actually usable here. Nothing selects it
 * automatically (priority sits below SIMD), so this is the test that keeps it
 * honest.
 */
class OpenClAsianParityTest {

  private static final int FIXINGS = 50;
  private static final double S0 = 100.0, K = 100.0, VOL = 0.20, RATE = 0.03, T = 1.0;
  private static final long PATHS = 200_000L;
  private static final long SEED = 0xA51A17L;

  /** GBM path in price space, arithmetic average of the fixings, discounted call payoff. */
  private static void asianCall(AadRecorder rec) {
    SDouble spot = rec.input("spot", S0);
    SDouble vol = rec.input("vol", VOL);
    double dt = T / FIXINGS;

    SDouble driftTerm = vol.mul(vol).mul(-0.5 * dt).add(RATE * dt);   // (r - vol^2/2) dt
    SDouble diffusion = vol.mul(Math.sqrt(dt));                       // vol sqrt(dt)

    SDouble price = spot;
    SDouble sum = rec.constant(0.0);
    for (int i = 0; i < FIXINGS; i++) {
      price = price.mul(driftTerm.add(diffusion.mul(rec.randn())).exp());
      sum = sum.add(price);
    }
    SDouble average = sum.mul(1.0 / FIXINGS);
    SDouble payoff = average.sub(K).max(0.0).mul(Math.exp(-RATE * T));
    rec.output(payoff);
  }

  @Test
  void openClMatchesCpuJitAndIsStable() {
    AadOptions fp64 = new AadOptions(AadOptions.Precision.FLOAT64, true);

    AadEngine openCl = AadEngines.find("opencl", fp64).orElse(null);
    assumeTrue(openCl != null, "no usable OpenCL engine on this machine");
    System.out.println("OpenCL engine: " + openCl.describe());

    AadTape tape = AadRecorder.record(OpenClAsianParityTest::asianCall);
    AadEngine oracleEngine = AadEngines.find("cpu-jit", fp64)
        .orElseGet(() -> AadEngines.require("cpu", fp64));

    AadResult oracle;
    try (var exe = oracleEngine.compile(tape, fp64)) {
      oracle = exe.replaySafe(PATHS, SEED);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    AadResult first;
    double compileSeconds;
    try (var exe = openCl.compile(tape, fp64)) {
      compileSeconds = exe.compileSeconds();
      first = exe.replay(PATHS, 0L, SEED);

      // Stability: 40 back-to-back dispatches on the device, fixed (paths, seed).
      for (int i = 0; i < 40; i++) {
        AadResult r = exe.replay(PATHS, 0L, SEED);
        assertEquals(first.value(), r.value(), 0.0, "OpenCL replay #" + i + " drifted");
        assertEquals(first.gradient("spot"), r.gradient("spot"), 0.0, "delta drift #" + i);
        assertEquals(first.gradient("vol"), r.gradient("vol"), 0.0, "vega drift #" + i);
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    System.out.printf(Locale.ROOT,
        "%-8s price=%.6f  delta=%.6f  vega=%.6f%n", "cpu", oracle.value(),
        oracle.gradient("spot"), oracle.gradient("vol"));
    System.out.printf(Locale.ROOT,
        "%-8s price=%.6f  delta=%.6f  vega=%.6f   (compile %.2fs)%n", "opencl", first.value(),
        first.gradient("spot"), first.gradient("vol"), compileSeconds);

    double priceTol = 5e-4 + 5e-4 * Math.abs(oracle.value());
    assertEquals(oracle.value(), first.value(), priceTol, "Asian price vs cpu oracle");
    assertEquals(oracle.gradient("spot"), first.gradient("spot"),
        5e-3 + 5e-3 * Math.abs(oracle.gradient("spot")), "delta vs cpu oracle");
    assertEquals(oracle.gradient("vol"), first.gradient("vol"),
        5e-3 + 5e-3 * Math.abs(oracle.gradient("vol")), "vega vs cpu oracle");

    assertTrue(first.value() > 0.0 && first.value() < S0, "sane Asian call premium");
  }
}
