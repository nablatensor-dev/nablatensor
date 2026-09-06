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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadEngines;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.Nabla;
import com.nablatensor.engine.SDouble;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@code rec.randu()} and named {@code rec.stream(...)}: the {@code cpu} oracle
 * and the {@code cpu-jit} kernel must agree bit-for-bit on the extended random
 * surface, uniforms must be a genuine {@code [0,1)} sample, named streams must be
 * independent, and a backend that has not grown native support must decline the
 * tape so selection falls back.
 */
@Tag("mc")
class RandomStreamsTest {

  private static final long N = 400_000L;
  private static final long SEED = 0xC0FFEEL;

  private static double meanOnEngine(String engine, java.util.function.Consumer<AadRecorder> body) {
    try (Nabla.Pricer p = Nabla.model(body).fp64().priceOnly().on(engine).build()) {
      return p.value().scenarios(N).seed(SEED).run().price();
    }
  }

  @Test
  void uniformDrawHasTheRightMeanAndRange() {
    // mean of one U(0,1) draw
    double mean = meanOnEngine("cpu", rec -> {
      rec.input("x", 0.0);          // a differentiable input so the tape is well-formed
      rec.output(rec.randu());
    });
    assertEquals(0.5, mean, 5e-3, "E[U(0,1)] ~ 0.5");

    // min over a batch: with 400k draws the smallest is well under 0.01 and never negative
    double minish = meanOnEngine("cpu", rec -> {
      rec.input("x", 0.0);
      SDouble u = rec.randu();
      rec.output(u.mul(u).mul(u).mul(u));   // E[U^4] = 1/5, and stays in [0,1)
    });
    assertEquals(0.2, minish, 5e-3, "E[U^4] ~ 1/5, so draws are in [0,1)");
  }

  @Test
  void cpuAndJitAgreeBitForBitOnUniformsAndNamedStreams() {
    java.util.function.Consumer<AadRecorder> body = rec -> {
      SDouble s0 = rec.input("S0", 100.0);
      SDouble vol = rec.input("sigma", 0.2);
      AadRecorder.RandomStream jumps = rec.stream("jump-clock");
      SDouble acc = s0;
      for (int t = 0; t < 16; t++) {
        SDouble z = rec.randn();                       // default stream, normal
        SDouble u = jumps.randu();                     // named stream, uniform
        SDouble jump = com.nablatensor.ops.Smooth.gt(rec, u, 0.97, 0.01);  // rare "jump"
        acc = acc.mul(vol.mul(z).mul(0.05).add(1.0)).add(jump.mul(2.0));
      }
      rec.output(acc);
    };
    double cpu = meanOnEngine("cpu", body);
    double jit = meanOnEngine("cpu-jit", body);
    assertEquals(cpu, jit, 0.0, "cpu oracle and cpu-jit kernel agree bit-for-bit on randu + named streams");
  }

  @Test
  void namedStreamsAreIndependent() {
    // E[ z_a * z_b ] over independent streams is ~0; over the same stream it would be ~1
    double crossDifferentStreams = meanOnEngine("cpu-jit", rec -> {
      rec.input("x", 0.0);
      SDouble za = rec.stream("a").randn();
      SDouble zb = rec.stream("b").randn();
      rec.output(za.mul(zb));
    });
    assertEquals(0.0, crossDifferentStreams, 5e-3, "independent streams: E[z_a z_b] ~ 0");
  }

  @Test
  void aBackendWithoutNativeSupportDeclinesSoSelectionFallsBack() {
    var tape = AadRecorder.record(rec -> {
      rec.input("x", 0.0);
      rec.output(rec.randu().add(rec.stream("k").randn()));
    });
    AadOptions fp64 = new AadOptions(AadOptions.Precision.FLOAT64, true);
    for (AadEngine e : AadEngines.discovered()) {
      if (e.name().equals("simd") && e.isAvailable()) {
        assertThrows(UnsupportedOperationException.class, () -> e.compile(tape, fp64),
            "simd must decline a randu/named-stream tape");
      }
    }
    // cpu-jit takes it
    try (Nabla.Pricer p = Nabla.model(rec -> {
          rec.input("x", 0.0);
          rec.output(rec.randu().add(rec.stream("k").randn()));
        }).fp64().priceOnly().on("cpu-jit").build()) {
      assertTrue(Double.isFinite(p.value().scenarios(N).seed(SEED).run().price()));
    }
  }

  @Test
  void singleStreamNormalTapeIsUnaffected() {
    java.util.function.Consumer<AadRecorder> body = rec -> {
      SDouble s0 = rec.input("S0", 100.0);
      SDouble vol = rec.input("sigma", 0.2);
      SDouble s = s0;
      for (int t = 0; t < 32; t++) {
        s = s.mul(vol.mul(rec.randn()).mul(0.03).add(1.0));
      }
      rec.output(s);
    };
    assertEquals(meanOnEngine("cpu", body), meanOnEngine("cpu-jit", body), 0.0,
        "a plain single-stream normal tape is still bit-identical across cpu and cpu-jit");
    assertNotEquals(0.0, meanOnEngine("cpu", body));
  }
}
