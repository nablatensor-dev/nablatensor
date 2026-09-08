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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.nablatensor.engine.AadEngines;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.SDouble;
import java.util.List;
import java.util.Locale;
import org.junit.jupiter.api.Test;

/**
 * Ablation of the two performance-relevant knobs the OpenCL GPU replay engine
 * keeps — the {@code -cl-mad-enable} build flag and the {@code native_*}
 * transcendentals — measured in one JVM against a {@code cpu-jit} fp64 oracle on
 * the same seed. Every configuration must stay bit-identical run-to-run and
 * within Monte-Carlo tolerance of the oracle; the table it prints shows what
 * each buys.
 *
 * <p>Skips unless OpenCL is usable (nothing selects it automatically).
 */
class OpenClSpeedupAblationTest {

  private static final int FIXINGS = 50;
  private static final double S0 = 100.0, K = 100.0, VOL = 0.20, RATE = 0.03, T = 1.0;
  private static final long PATHS = 1_000_000L;
  private static final long SEED = 0xA51A17L;
  private static final int WARMUP = 30;
  private static final int RUNS = 40;

  private static void asianCall(AadRecorder rec) {
    SDouble spot = rec.input("spot", S0);
    SDouble vol = rec.input("vol", VOL);
    double dt = T / FIXINGS;
    SDouble driftTerm = vol.mul(vol).mul(-0.5 * dt).add(RATE * dt);
    SDouble diffusion = vol.mul(Math.sqrt(dt));
    SDouble price = spot;
    SDouble sum = rec.constant(0.0);
    for (int i = 0; i < FIXINGS; i++) {
      price = price.mul(driftTerm.add(diffusion.mul(rec.randn())).exp());
      sum = sum.add(price);
    }
    rec.output(sum.mul(1.0 / FIXINGS).sub(K).max(0.0).mul(Math.exp(-RATE * T)));
  }

  private record Config(String name, String buildOptions, boolean nativeMath) {
  }

  private record Result(Config cfg, double settledMs, double priceBp, boolean bitExact) {
  }

  @Test
  void ablateKeptOpenClSpeedups() {
    AadOptions fp64 = new AadOptions(AadOptions.Precision.FLOAT64, true);
    AadOptions fp32 = new AadOptions(AadOptions.Precision.FLOAT32, true);
    assumeTrue(AadEngines.find("opencl", fp32).isPresent(), "no usable OpenCL engine");

    AadTape tape = AadRecorder.record(OpenClSpeedupAblationTest::asianCall);
    AadResult oracle;
    try (var exe = AadEngines.find("cpu-jit", fp64).orElseGet(() -> AadEngines.require("cpu", fp64))
        .compile(tape, fp64)) {
      oracle = exe.replaySafe(4_000_000L, SEED);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    List<Config> configs = List.of(
        new Config("baseline (no flags, accurate math)", "", false),
        new Config("+ -cl-mad-enable", "-cl-mad-enable", false),
        new Config("+ native_* transcendentals", "", true),
        new Config("kept default (-cl-mad-enable + native_*)", "-cl-mad-enable", true));

    Result[] results = new Result[configs.size()];
    for (int i = 0; i < configs.size(); i++) {
      results[i] = measure(configs.get(i), tape, fp32, oracle);
    }

    System.out.println();
    System.out.printf(Locale.ROOT,
        "OpenCL GPU fp32 kept-speedup ablation · Asian call · %d fixings · %d paths/dispatch%n",
        FIXINGS, PATHS);
    System.out.printf(Locale.ROOT, "oracle (cpu-jit fp64): price %.6f%n%n", oracle.value());
    System.out.println("| configuration | settled ms/1M | throughput | speed-up vs baseline | Δ price | run-to-run |");
    System.out.println("|---|---|---|---|---|---|");
    double base = results[0].settledMs();
    for (Result r : results) {
      System.out.printf(Locale.ROOT, "| %s | %.2f | %.2e scen/s | %.2f× | %+.2f bp | %s |%n",
          r.cfg().name(), r.settledMs(), PATHS / (r.settledMs() / 1e3), base / r.settledMs(),
          r.priceBp(), r.bitExact() ? "bit-exact" : "**drifts**");
    }
    System.out.println();

    for (Result r : results) {
      assertEquals(oracle.value(), oracle.value() * (1 + r.priceBp() / 1e4),
          Math.abs(oracle.value()) * 3e-3, r.cfg().name() + ": fp32 price left tolerance");
    }
  }

  private static Result measure(Config cfg, AadTape tape, AadOptions fp32, AadResult oracle) {
    System.setProperty("nablatensor.opencl.buildOptions", cfg.buildOptions());
    System.setProperty("nablatensor.opencl.nativeMath", Boolean.toString(cfg.nativeMath()));
    try {
      try (var exe = AadEngines.require("opencl", fp32).compile(tape, fp32)) {
        for (int i = 0; i < WARMUP; i++) {
          exe.replay(PATHS, 0L, SEED);
        }
        AadResult baseline = exe.replay(PATHS, 0L, SEED);
        boolean bitExact = true;
        double[] ms = new double[RUNS];
        for (int i = 0; i < RUNS; i++) {
          long s = System.nanoTime();
          AadResult r = exe.replay(PATHS, 0L, SEED);
          ms[i] = (System.nanoTime() - s) / 1e6;
          if (Double.compare(r.value(), baseline.value()) != 0
              || Double.compare(r.gradient("spot"), baseline.gradient("spot")) != 0
              || Double.compare(r.gradient("vol"), baseline.gradient("vol")) != 0) {
            bitExact = false;
          }
        }
        java.util.Arrays.sort(ms);
        double bp = 1e4 * (baseline.value() - oracle.value()) / oracle.value();
        return new Result(cfg, ms[RUNS / 2], bp, bitExact);
      }
    } catch (RuntimeException | LinkageError e) {
      throw new RuntimeException(cfg.name() + " failed: " + e.getMessage(), e);
    } finally {
      System.clearProperty("nablatensor.opencl.buildOptions");
      System.clearProperty("nablatensor.opencl.nativeMath");
    }
  }
}
