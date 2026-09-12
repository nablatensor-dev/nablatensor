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

import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Products;
import java.util.Locale;

/**
 * Checkpointed vs. plain (fully-unrolled) adjoint kernel on one GPU engine,
 * same tape, same seed, one process per mode. Checkpointing on {@code cuda},
 * {@code rocm} and {@code opencl} is off unless
 * {@code -Dnablatensor.checkpoint.minNodes=<n>} names a threshold below the
 * tape's node count; {@code vulkan} checkpoints automatically above 768 nodes
 * unless {@code -Dnablatensor.vulkan.ckpt=1} forces it off. Prints one
 * {@code RESULT ...} line meant to be parsed by the calling notebook/script.
 *
 * <p>Run: {@code mvn -q -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.bench.CheckpointBench
 * -Dengine=cuda -Dsteps=252 -Dscenarios=1000000
 * -Dnablatensor.checkpoint.minNodes=100 -Dmode=checkpointed [-Dgreeks=false]}
 */
public final class CheckpointBench {

  private CheckpointBench() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    int steps = Integer.getInteger("steps", 252);
    long scenarios = Long.getLong("scenarios", 1_000_000L);
    long seed = Long.getLong("seed", 42L);
    int warmup = Integer.getInteger("warmup", 20);
    int repeat = Integer.getInteger("repeat", 25);
    String engine = System.getProperty("engine", "cuda");
    String mode = System.getProperty("mode", "unspecified");
    boolean greeks = Boolean.parseBoolean(System.getProperty("greeks", "true"));

    System.out.printf(Locale.ROOT, "# NablaTensor checkpoint bench%n%n");
    System.out.printf(Locale.ROOT, "- machine   : JDK %s, %s %s%n",
        Runtime.version(), System.getProperty("os.name"), System.getProperty("os.arch"));
    System.out.printf(Locale.ROOT, "- engine    : %s   mode: %s%n", engine, mode);
    System.out.printf(Locale.ROOT, "- product   : Asian call, %d fixings, fp32, %s%n",
        steps, greeks ? "price + Greeks" : "price only");
    System.out.printf(Locale.ROOT, "- scenarios : %,d   seed : %d   warmup: %d   repeat: %d%n%n",
        scenarios, seed, warmup, repeat);

    var builder = MonteCarlo.of(Products.asianCall()).market(market).steps(steps).fp32();
    builder = greeks ? builder.greeks() : builder.priceOnly();
    try (MonteCarlo<EquityMarket> mc = builder.on(engine).build()) {
      System.out.printf(Locale.ROOT, "nodes: %d%n", mc.nodes());

      for (int i = 0; i < warmup; i++) {
        mc.run(scenarios, seed);
      }
      Nabla.TypedValuation<EquityMarket> baseline = mc.run(scenarios, seed);

      double bestSeconds = Double.MAX_VALUE;
      boolean bitExact = true;
      for (int i = 0; i < repeat; i++) {
        Nabla.TypedValuation<EquityMarket> r = mc.run(scenarios, seed);
        bestSeconds = Math.min(bestSeconds, r.seconds());
        if (Double.compare(r.price(), baseline.price()) != 0) {
          bitExact = false;
        }
      }

      double delta = greeks ? baseline.greek(EquityMarket::spot) : Double.NaN;
      System.out.printf(Locale.ROOT,
          "RESULT engine=%s mode=%s nodes=%d settled_s=%.6f scen_per_s=%.3e "
              + "price=%.6f delta=%.6f bit_exact=%b%n",
          engine, mode, mc.nodes(), bestSeconds, scenarios / bestSeconds,
          baseline.price(), delta, bitExact);
    }
  }
}
