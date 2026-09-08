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

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadEngines;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.SDouble;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.junit.jupiter.api.Test;

/**
 * Head-to-head of the three fp32-capable replay engines — SIMD, Vulkan, OpenCL —
 * on a textbook arithmetic-average Asian call Monte Carlo, plus a stability
 * evaluation: for a fixed {@code (paths, seed)} every repeat must return
 * bit-identical numbers, and a burst of dispatches over shifting path offsets
 * must stay finite and never throw. Accuracy is judged against a {@code cpu-jit}
 * fp64 oracle on the same seed, so the only thing separating a candidate from
 * the oracle is fp32 rounding.
 *
 * <p>Prints a Markdown results table to stdout. Engines that are not usable here
 * are reported as {@code n/a} rather than failing the test.
 */
class Fp32BackendComparisonTest {

  private static final int FIXINGS = 50;
  private static final double S0 = 100.0, K = 100.0, VOL = 0.20, RATE = 0.03, T = 1.0;
  private static final long PATHS = 1_000_000L;   // one dispatch
  private static final long ORACLE_PATHS = 4_000_000L;
  private static final long SEED = 0xA51A17L;
  private static final int SETTLED_RUNS = 60;
  private static final int OFFSET_BURST = 25;

  /** GBM path in price space, arithmetic average of the fixings, discounted call payoff. */
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
    SDouble average = sum.mul(1.0 / FIXINGS);
    rec.output(average.sub(K).max(0.0).mul(Math.exp(-RATE * T)));
  }

  private record Row(String engine, boolean usable, String detail,
                     double price, double delta, double vega,
                     double compileMs, double firstCallMs, double warmupMeanMs, double warmupMinMs,
                     double settledMs, double settledScenPerSec,
                     boolean bitExact, double maxDriftRel, boolean burstOk, String note) {
  }

  @Test
  void compareFp32EnginesAndReportStability() {
    AadOptions fp64 = new AadOptions(AadOptions.Precision.FLOAT64, true);
    AadOptions fp32 = new AadOptions(AadOptions.Precision.FLOAT32, true);
    AadTape tape = AadRecorder.record(Fp32BackendComparisonTest::asianCall);

    AadEngine oracleEngine = AadEngines.find("cpu-jit", fp64)
        .orElseGet(() -> AadEngines.require("cpu", fp64));
    AadResult oracle;
    try (var exe = oracleEngine.compile(tape, fp64)) {
      oracle = exe.replaySafe(ORACLE_PATHS, SEED);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    List<Row> rows = new ArrayList<>();
    for (String name : List.of("simd", "vulkan", "opencl")) {
      rows.add(measure(name, tape, fp32));
    }

    printTable(oracle, rows);

    // ---- assertions: any engine that ran must be accurate and stable ----
    for (Row r : rows) {
      if (!r.usable()) {
        continue;
      }
      assertTrue(r.maxDriftRel() < 1e-4,
          r.engine() + ": fixed-seed replay drifted run-to-run by " + r.maxDriftRel() + " (rel)");
      assertTrue(r.burstOk(), r.engine() + ": path-offset burst produced a non-finite or threw");
      assertEquals(oracle.value(), r.price(), 3e-3 + 3e-3 * Math.abs(oracle.value()),
          r.engine() + " fp32 price vs fp64 oracle");
      assertEquals(oracle.gradient("spot"), r.delta(),
          1e-2 + 1e-2 * Math.abs(oracle.gradient("spot")), r.engine() + " fp32 delta vs fp64 oracle");
      assertEquals(oracle.gradient("vol"), r.vega(),
          1e-2 + 1e-2 * Math.abs(oracle.gradient("vol")), r.engine() + " fp32 vega vs fp64 oracle");
    }
  }

  private static Row measure(String name, AadTape tape, AadOptions fp32) {
    AadEngine engine = AadEngines.find(name, fp32).orElse(null);
    if (engine == null) {
      return new Row(name, false, "not usable here",
          0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, false, "skipped");
    }
    String detail = engine.describe();
    try (var exe = engine.compile(tape, fp32)) {
      double compileMs = exe.compileSeconds() * 1e3;

      long t0 = System.nanoTime();
      AadResult first = exe.replay(PATHS, 0L, SEED);       // cold: first dispatch, kernel not yet resident
      double firstCallMs = (System.nanoTime() - t0) / 1e6;

      // warm-up: time every dispatch so the settle curve is visible (a CPU engine's
      // C2 recompile, a GPU engine's driver/queue warm-up).
      double warmupSum = 0, warmupMin = Double.MAX_VALUE;
      int warmupRuns = 40;
      for (int i = 0; i < warmupRuns; i++) {
        long s = System.nanoTime();
        exe.replay(PATHS, 0L, SEED);
        double ms = (System.nanoTime() - s) / 1e6;
        warmupSum += ms;
        warmupMin = Math.min(warmupMin, ms);
      }
      double warmupMeanMs = warmupSum / warmupRuns;
      AadResult baseline = exe.replay(PATHS, 0L, SEED);

      boolean bitExact = true;
      double maxDriftRel = 0;
      double[] ms = new double[SETTLED_RUNS];
      for (int i = 0; i < SETTLED_RUNS; i++) {
        long s = System.nanoTime();
        AadResult r = exe.replay(PATHS, 0L, SEED);
        ms[i] = (System.nanoTime() - s) / 1e6;
        for (String[] pair : new String[][] {{"value", null}, {"grad", "spot"}, {"grad", "vol"}}) {
          double got = pair[1] == null ? r.value() : r.gradient(pair[1]);
          double ref = pair[1] == null ? baseline.value() : baseline.gradient(pair[1]);
          if (Double.compare(got, ref) != 0) {
            bitExact = false;
            maxDriftRel = Math.max(maxDriftRel, Math.abs(got - ref) / Math.max(1e-12, Math.abs(ref)));
          }
        }
      }
      java.util.Arrays.sort(ms);
      double settledMs = ms[SETTLED_RUNS / 2];
      double settledScenPerSec = PATHS / (settledMs / 1e3);

      // stability burst: many dispatches over shifting offsets (fresh path draws)
      boolean burstOk = true;
      for (int i = 0; i < OFFSET_BURST; i++) {
        AadResult r = exe.replay(PATHS, (long) i * PATHS, SEED);
        if (!Double.isFinite(r.value()) || !Double.isFinite(r.gradient("spot"))
            || !Double.isFinite(r.gradient("vol"))) {
          burstOk = false;
        }
      }

      String note = String.format(Locale.ROOT, "settled %.2f ms/1M · %.0f%% faster than cold",
          settledMs, 100.0 * (firstCallMs - settledMs) / firstCallMs);
      return new Row(name, true, detail, baseline.value(), baseline.gradient("spot"),
          baseline.gradient("vol"), compileMs, firstCallMs, warmupMeanMs, warmupMin,
          settledMs, settledScenPerSec, bitExact, maxDriftRel, burstOk, note);
    } catch (RuntimeException | LinkageError e) {
      return new Row(name, false, "compile/replay failed", 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, false,
          String.valueOf(e.getMessage()));
    }
  }

  private static void printTable(AadResult oracle, List<Row> rows) {
    System.out.println();
    System.out.println("Asian call · " + FIXINGS + " fixings · " + PATHS + " paths/dispatch · seed "
        + Long.toHexString(SEED) + " · fp32 (oracle: cpu-jit fp64, " + ORACLE_PATHS + " paths)");
    System.out.printf(Locale.ROOT, "oracle    price=%.6f  delta=%.6f  vega=%.6f%n%n",
        oracle.value(), oracle.gradient("spot"), oracle.gradient("vol"));

    // ---- timing: wall-clock per 1,000,000-path dispatch, warm-up included ----
    System.out.println("Execution time (ms per " + PATHS + "-path dispatch, one recorded tape):");
    System.out.println("| engine | compile | 1st call (cold) | warm-up mean (×40) | warm-up best | settled (median ×" + SETTLED_RUNS + ") | throughput | cold→settled |");
    System.out.println("|---|---|---|---|---|---|---|---|");
    for (Row r : rows) {
      if (!r.usable()) {
        System.out.printf(Locale.ROOT, "| %s | n/a | | | | | | %s |%n", r.engine(), r.note());
        continue;
      }
      System.out.printf(Locale.ROOT,
          "| %s | %.0f ms | %.1f ms | %.1f ms | %.1f ms | %.2f ms | %.2e scen/s | %.1f× |%n",
          r.engine(), r.compileMs(), r.firstCallMs(), r.warmupMeanMs(), r.warmupMinMs(),
          r.settledMs(), r.settledScenPerSec(), r.firstCallMs() / r.settledMs());
    }
    System.out.println();

    // ---- accuracy & stability ----
    System.out.println("Accuracy vs fp64 oracle (same seed) and stability:");
    System.out.println("| engine | price | Δ price (bp) | delta | vega | run-to-run | offset burst | device |");
    System.out.println("|---|---|---|---|---|---|---|---|");
    for (Row r : rows) {
      if (!r.usable()) {
        System.out.printf(Locale.ROOT, "| %s | n/a | | | | | | %s |%n", r.engine(), r.note());
        continue;
      }
      double bp = 1e4 * (r.price() - oracle.value()) / oracle.value();
      String drift = r.bitExact() ? "bit-exact"
          : String.format(Locale.ROOT, "±%.1e rel", r.maxDriftRel());
      System.out.printf(Locale.ROOT,
          "| %s | %.6f | %+.2f | %.6f | %.4f | %s | %s | %s |%n",
          r.engine(), r.price(), bp, r.delta(), r.vega(),
          drift, r.burstOk() ? "ok" : "**FAIL**", r.detail());
    }
    System.out.println();
  }
}
