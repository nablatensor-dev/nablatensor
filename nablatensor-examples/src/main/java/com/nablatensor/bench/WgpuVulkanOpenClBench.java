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
package com.nablatensor.bench;

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadEngines;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Products;
import java.util.List;
import java.util.Locale;

/**
 * GPU-shaped replay engines, same tape, same seed, each pinned with
 * {@code .on(name)} so a backend that {@code .fastest()} would never pick
 * (wgpu sits just below Vulkan) still gets measured. Vulkan and wgpu are
 * single-precision only, so both run {@code .fp32()}.
 *
 * <p>OpenCL is skipped by default: {@code OpenClAadEngine} already documents
 * that its default device "shares the display scheduler and carries the same
 * wedge risk as the ROCm path" on an integrated GPU, and sustained compute
 * through it has been observed to wedge and hard-reset the device on at least
 * one such machine. Pass {@code -Dinclude.opencl=true} to opt in; it then
 * runs {@code .fp64()} (its replay kernel always accumulates in
 * {@code double}), and the table says so per row rather than implying a
 * same-precision comparison it is not.
 *
 * <p>Reports the cold call (includes shader/kernel compile) separately from
 * the settled best-of-N rate: wgpu's WGSL is compiled by wgpu-native's own
 * WGSL→SPIR-V frontend (naga) rather than {@code libshaderc}'s GLSL compiler,
 * and that shows up almost entirely in the cold number, not the settled one.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.bench.WgpuVulkanOpenClBench
 * -Dscenarios=2000000 -Dsteps=252}
 */
public final class WgpuVulkanOpenClBench {

  private WgpuVulkanOpenClBench() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    int steps = Integer.getInteger("steps", 252);
    long scenarios = Long.getLong("scenarios", 2_000_000L);
    long seed = Long.getLong("seed", 42L);
    boolean includeOpenCl = Boolean.getBoolean("include.opencl");

    System.out.printf(Locale.ROOT, "# wgpu vs Vulkan%s%n%n", includeOpenCl ? " vs OpenCL" : "");
    System.out.printf(Locale.ROOT, "- machine   : JDK %s, %s %s, %d processors%n",
        Runtime.version(), System.getProperty("os.name"), System.getProperty("os.arch"),
        Runtime.getRuntime().availableProcessors());
    System.out.printf(Locale.ROOT, "- product   : Asian call, %d fixings%n", steps);
    System.out.printf(Locale.ROOT, "- scenarios : %,d   seed : %d%n%n", scenarios, seed);

    System.out.printf(Locale.ROOT, "| engine | precision | price | delta | cold (1st call) | settled (best of 3) | scenarios/s | runs on |%n");
    System.out.printf(Locale.ROOT, "|---|---|--:|--:|--:|--:|--:|---|%n");

    row("vulkan", true, market, steps, scenarios, seed);
    row("wgpu", true, market, steps, scenarios, seed);
    if (includeOpenCl) {
      row("opencl", false, market, steps, scenarios, seed);
    } else {
      System.out.printf(Locale.ROOT, "| opencl | fp64 | — | — | — | — | — | skipped (wedge risk on this GPU; pass -Dinclude.opencl=true) |%n");
    }
    System.out.println();
  }

  private static void row(String engine, boolean fp32, EquityMarket market, int steps, long scenarios, long seed) {
    AadOptions probe = new AadOptions(fp32 ? AadOptions.Precision.FLOAT32 : AadOptions.Precision.FLOAT64, true);
    List<AadEngine> found = AadEngines.available(probe).stream().filter(e -> e.name().equals(engine)).toList();
    if (found.isEmpty()) {
      System.out.printf(Locale.ROOT, "| %s | %s | — | — | — | — | — | not available |%n",
          engine, fp32 ? "fp32" : "fp64");
      return;
    }
    AadEngine e = found.get(0);
    MonteCarlo.Builder<EquityMarket> builder = MonteCarlo.of(Products.asianCall()).market(market).steps(steps);
    builder = fp32 ? builder.fp32() : builder.fp64();
    try (MonteCarlo<EquityMarket> greeks = builder.greeks().on(engine).build()) {
      long coldStart = System.nanoTime();
      Nabla.TypedValuation<EquityMarket> first = greeks.run(scenarios, seed);
      double coldSeconds = (System.nanoTime() - coldStart) / 1e9;

      double settled = Double.MAX_VALUE;
      Nabla.TypedValuation<EquityMarket> last = first;
      for (int i = 0; i < 3; i++) {
        last = greeks.run(scenarios, seed);
        settled = Math.min(settled, last.seconds());
      }
      System.out.printf(Locale.ROOT, "| %s | %s | %.5f | %.5f | %.4f s | %.4f s | %.2e | %s |%n",
          engine, fp32 ? "fp32" : "fp64", last.price(), last.greek(EquityMarket::spot),
          coldSeconds, settled, last.scenariosPerSecond(), e.describe());
    } catch (RuntimeException | LinkageError ex) {
      System.out.printf(Locale.ROOT, "| %s | %s | — | — | — | — | — | failed: %s |%n",
          engine, fp32 ? "fp32" : "fp64", ex.getMessage());
    }
  }
}
