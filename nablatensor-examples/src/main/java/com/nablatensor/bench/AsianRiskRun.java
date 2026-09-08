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
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.Products;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The plan's headline risk run: an arithmetic Asian call with Greeks, over
 * {@code 1_000_000 x 10_000 = 1e10} scenarios.
 *
 * <p>One tape is recorded once; every backend this machine can run replays it.
 * Each backend is measured on a probe batch (best-of-3) and the {@code 1e10}
 * wall clock is the linear projection {@code total / (probe / probeSeconds)} —
 * the run itself would be watchdog-chunked, not one launch.
 *
 * <p>Run:
 * <pre>{@code
 * MAVEN_OPTS="--add-modules jdk.incubator.vector" mvn -q -o -pl nablatensor-bench exec:java \
 *   -Dexec.mainClass=com.nablatensor.bench.AsianRiskRun \
 *   -Dprobe=2000000 -Dsteps=252
 * }</pre>
 */
public final class AsianRiskRun {

  private AsianRiskRun() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    int steps = Integer.getInteger("steps", 252);
    long probe = Long.getLong("probe", 2_000_000L);
    long seed = Long.getLong("seed", 42L);
    long total = Long.getLong("total", 1_000_000L * 10_000L);   // 1e10

    System.out.printf(Locale.ROOT, "# NablaTensor — Asian risk run  (%,d x %,d = %.0e scenarios)%n%n",
        1_000_000L, 10_000L, (double) total);
    System.out.printf(Locale.ROOT, "- machine : JDK %s, %s %s, %d processors%n",
        Runtime.version(), System.getProperty("os.name"), System.getProperty("os.arch"),
        Runtime.getRuntime().availableProcessors());
    System.out.printf(Locale.ROOT, "- product : arithmetic Asian call, %d fixings, GBM, ATM 1y%n", steps);
    System.out.printf(Locale.ROOT, "- output  : price + delta/vega/rho/dV/dK/dV/dT from one adjoint sweep%n");
    System.out.printf(Locale.ROOT, "- probe   : %,d scenarios, best of 3, seed %d%n%n", probe, seed);

    for (boolean fp32 : new boolean[] {false, true}) {
      System.out.printf(Locale.ROOT, "## %s%n%n", fp32 ? "fp32" : "fp64");
      System.out.printf(Locale.ROOT, "| engine | build | nodes | price | delta | scen/s | 1e10 wall clock | vs cpu |%n");
      System.out.printf(Locale.ROOT, "|---|--:|--:|--:|--:|--:|--:|--:|%n");

      double cpuThroughput = Double.NaN;
      for (AadEngine engine : available(fp32)) {
        String name = engine.name();
        try (MonteCarlo<EquityMarket> mc = configure(MonteCarlo.of(Products.asianCall())
            .market(market).steps(steps).greeks().on(name), fp32).build()) {

          mc.run(probe, seed);                          // warm (build + C2)
          mc.run(probe, seed);
          Nabla.TypedValuation<EquityMarket> best = null;
          for (int i = 0; i < 3; i++) {
            Nabla.TypedValuation<EquityMarket> p = mc.run(probe, seed);
            if (best == null || p.seconds() < best.seconds()) {
              best = p;
            }
          }
          double throughput = probe / best.seconds();
          if (name.equals("cpu")) {
            cpuThroughput = throughput;
          }
          double projected = total / throughput;
          String speedup = Double.isNaN(cpuThroughput) ? "-"
              : String.format(Locale.ROOT, "%.1fx", throughput / cpuThroughput);

          System.out.printf(Locale.ROOT, "| %-9s | %5.0f ms | %d | %.5f | %.5f | %.2e | %s | %s |%n",
              name, mc.buildSeconds() * 1e3, mc.nodes(),
              best.price(), best.greek(EquityMarket::spot), throughput, wall(projected), speedup);
        } catch (RuntimeException | LinkageError e) {
          System.out.printf(Locale.ROOT, "| %-9s | skipped: %s |%n", name, e.getMessage());
        }
      }
      System.out.println();
    }

    System.out.println("Projection is linear from the probe; the real 1e10 run is watchdog-chunked");
    System.out.println("(replaySafe) and would reuse the same compiled kernel and draw cache.");
  }

  private static MonteCarlo.Builder<EquityMarket> configure(MonteCarlo.Builder<EquityMarket> b, boolean fp32) {
    return fp32 ? b.fp32() : b.fp64();
  }

  private static List<AadEngine> available(boolean fp32) {
    AadOptions options = new AadOptions(
        fp32 ? AadOptions.Precision.FLOAT32 : AadOptions.Precision.FLOAT64, true);
    // cpu first so the speedup column has its baseline.
    List<AadEngine> ordered = new ArrayList<>();
    for (AadEngine e : AadEngines.available(options)) {
      if (e.name().equals("cpu")) {
        ordered.add(0, e);
      } else {
        ordered.add(e);
      }
    }
    return ordered;
  }

  private static String wall(double seconds) {
    if (seconds < 90) {
      return String.format(Locale.ROOT, "%.1f s", seconds);
    }
    if (seconds < 5400) {
      return String.format(Locale.ROOT, "%.1f min", seconds / 60);
    }
    return String.format(Locale.ROOT, "%.2f h", seconds / 3600);
  }
}
