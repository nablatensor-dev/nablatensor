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
package com.nablatensor.ops;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.nablatensor.engine.AadEngines;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.SDouble;
import com.nablatensor.engine.Nabla;
import java.util.function.BiFunction;
import org.junit.jupiter.api.Test;

/**
 * The op vocabulary is a deterministic function of its input, so a one-scenario
 * replay returns {@code f(x)} exactly and the adjoint returns {@code f'(x)}
 * exactly — no Monte-Carlo error to allow for.
 */
class OpsTest {

  record P(double x) {}

  /** Records {@code out = body(x)}, replays once, returns {@code {value, d/dx}}. */
  private static double[] eval(String engine, double x, BiFunction<Nabla.Inputs<P>, com.nablatensor.engine.AadRecorder, SDouble> body) {
    try (Nabla.TypedPricer<P> pricer = Nabla.model(new P(x), (rec, in) -> rec.output(body.apply(in, rec)))
        .fp64().greeks().on(engine).build()) {
      Nabla.TypedValuation<P> v = pricer.value().with(new P(x)).scenarios(1).seed(1L).run();
      return new double[] {v.price(), v.greek(P::x)};
    }
  }

  private static double sigmoid(double z) {
    return 1.0 / (1.0 + Math.exp(-z));
  }

  private static double refN(double x) {  // erfc-based reference, ~1e-7
    double z = Math.abs(x) / Math.sqrt(2.0);
    double t = 1.0 / (1.0 + 0.5 * z);
    double ans = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196
        + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398
        + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    double erfc = x >= 0 ? ans : 2.0 - ans;
    return 1.0 - 0.5 * erfc;
  }

  @Test
  void smoothStepValueAndAdjoint() {
    double w = 0.1;
    double x = 0.3;
    double[] r = eval("cpu-jit", x, (in, rec) -> Smooth.step(rec, in.of(P::x), w));
    double s = sigmoid(x / w);
    assertEquals(s, r[0], 1e-9, "sigmoid value");
    assertEquals(s * (1 - s) / w, r[1], 1e-8, "sigmoid derivative");
  }

  @Test
  void smoothStepApproachesTheDiscontinuousLimit() {
    for (double w : new double[] {0.1, 0.01, 0.001}) {
      double hi = eval("cpu-jit", 1.0, (in, rec) -> Smooth.step(rec, in.of(P::x), w))[0];
      double lo = eval("cpu-jit", -1.0, (in, rec) -> Smooth.step(rec, in.of(P::x), w))[0];
      assertTrue(hi > 1.0 - 10 * w, "step(+1) -> 1 as width shrinks, got " + hi + " at w=" + w);
      assertTrue(lo < 10 * w, "step(-1) -> 0 as width shrinks, got " + lo + " at w=" + w);
    }
  }

  @Test
  void rampIsSmoothedMax() {
    double w = 0.02;
    double[] pos = eval("cpu-jit", 0.5, (in, rec) -> Smooth.ramp(rec, in.of(P::x), w));
    double[] neg = eval("cpu-jit", -0.5, (in, rec) -> Smooth.ramp(rec, in.of(P::x), w));
    assertEquals(0.5, pos[0], 1e-3, "ramp(+) ~ x");
    assertEquals(0.0, neg[0], 1e-3, "ramp(-) ~ 0");
    assertTrue(pos[1] > 0.99 && pos[1] <= 1.0 + 1e-9, "ramp'(+) ~ 1");
    assertTrue(neg[1] >= 0.0 && neg[1] < 0.01, "ramp'(-) ~ 0");
  }

  @Test
  void normCdfMatchesReferenceAndItsOwnPdf() {
    for (double x : new double[] {-2.0, -0.5, 0.0, 0.7, 1.5}) {
      double[] r = eval("cpu-jit", x, (in, rec) -> SpecialFn.normCdf(rec, in.of(P::x)));
      assertEquals(refN(x), r[0], 2e-4, "N(" + x + ")");
      double pdf = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
      assertEquals(pdf, r[1], 3e-3, "N'(" + x + ") ~ phi(x)");
    }
  }

  @Test
  void powConstantExponent() {
    double p = 1.7;
    double x = 2.3;
    double[] r = eval("cpu-jit", x, (in, rec) -> SpecialFn.pow(in.of(P::x), p));
    assertEquals(Math.pow(x, p), r[0], 1e-9, "x^p");
    assertEquals(p * Math.pow(x, p - 1), r[1], 1e-8, "d/dx x^p");
  }

  @Test
  void customOpRunsIdenticallyOnEveryCpuBackend() {
    CustomOp.registerUnary("myclamp", (rec, x) -> Smooth.between(rec, x, -1.0, 1.0, 0.05).mul(x));
    double x = 0.4;
    double[] jit = eval("cpu-jit", x, (in, rec) -> CustomOp.unary("myclamp").apply(rec, in.of(P::x)));
    double[] cpu = eval("cpu", x, (in, rec) -> CustomOp.unary("myclamp").apply(rec, in.of(P::x)));
    double[] simd = eval("simd", x, (in, rec) -> CustomOp.unary("myclamp").apply(rec, in.of(P::x)));
    assertEquals(cpu[0], jit[0], 1e-9, "value cpu vs cpu-jit");
    assertEquals(cpu[1], jit[1], 1e-9, "adjoint cpu vs cpu-jit");
    assertEquals(cpu[0], simd[0], 1e-9, "value cpu vs simd");
    assertEquals(cpu[1], simd[1], 1e-9, "adjoint cpu vs simd");
  }

  @Test
  void customOpRunsOnAGpuBackendWithNoEngineEdit() {
    // Phase-1 DoD: a user-registered custom op runs on cpu-jit AND a GPU backend.
    CustomOp.registerUnary("myclamp", (rec, x) -> Smooth.between(rec, x, -1.0, 1.0, 0.05).mul(x));
    String gpu = firstAvailable("vulkan", "rocm", "cuda");
    assumeTrue(gpu != null, "no GPU backend available on this machine");

    double x = 0.4;
    double[] ref = eval("cpu-jit", x, (in, rec) -> CustomOp.unary("myclamp").apply(rec, in.of(P::x)));
    double[] onGpu = evalFp32(gpu, x, (in, rec) -> CustomOp.unary("myclamp").apply(rec, in.of(P::x)));
    assertEquals(ref[0], onGpu[0], 3e-4, "custom-op value: cpu-jit vs " + gpu);
    assertEquals(ref[1], onGpu[1], 3e-4, "custom-op adjoint: cpu-jit vs " + gpu);
  }

  private static double[] evalFp32(String engine, double x,
      BiFunction<Nabla.Inputs<P>, com.nablatensor.engine.AadRecorder, SDouble> body) {
    try (Nabla.TypedPricer<P> pricer = Nabla.model(new P(x), (rec, in) -> rec.output(body.apply(in, rec)))
        .fp32().greeks().on(engine).build()) {
      Nabla.TypedValuation<P> v = pricer.value().with(new P(x)).scenarios(1024).seed(1L).run();
      return new double[] {v.price(), v.greek(P::x)};
    }
  }

  private static String firstAvailable(String... names) {
    var usable = AadEngines.available(new AadOptions(AadOptions.Precision.FLOAT32, true))
        .stream().map(e -> e.name()).toList();
    for (String n : names) {
      if (usable.contains(n)) {
        return n;
      }
    }
    return null;
  }
}
