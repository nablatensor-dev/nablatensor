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

import com.nablatensor.engine.Nabla;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Prices several named risk measures off the same market and seed.
 *
 * <p>The engine tape has a single output, so this builds one {@link MonteCarlo}
 * per metric and replays them at a common seed — the metrics are consistent
 * across the same scenarios, and each still gets its full adjoint gradient. It
 * costs N kernels and N replays; a single tape with multiple named outputs
 * (one forward sweep, N reverse seeds) is a planned engine feature that will
 * make this one launch.
 *
 * <pre>{@code
 * try (MultiMetric mm = MultiMetric.market(EquityMarket.atmOneYear()).steps(128)
 *         .metric("call",    Products.europeanCall())
 *         .metric("digital",  ExoticProducts.digitalCash(OptionType.CALL, 1.0, 1.0))
 *         .metric("barrierUO", ExoticProducts.barrier(OptionType.CALL,
 *                     ExoticProducts.Barrier.UP_OUT, 130.0, 1.0))
 *         .on("cpu-jit").build()) {
 *
 *   Map<String, Nabla.TypedValuation<EquityMarket>> r = mm.run(1_000_000, 42L);
 *   r.get("barrierUO").greek(EquityMarket::spot);
 * }
 * }</pre>
 */
public final class MultiMetric implements AutoCloseable {

  private final Map<String, MonteCarlo<EquityMarket>> kernels;

  private MultiMetric(Map<String, MonteCarlo<EquityMarket>> kernels) {
    this.kernels = kernels;
  }

  public static Builder market(EquityMarket market) {
    return new Builder(market);
  }

  /** Replays every metric at the same seed and scenario count. */
  public Map<String, Nabla.TypedValuation<EquityMarket>> run(long scenarios, long seed) {
    Map<String, Nabla.TypedValuation<EquityMarket>> out = new LinkedHashMap<>();
    kernels.forEach((name, mc) -> out.put(name, mc.run(scenarios, seed)));
    return out;
  }

  public java.util.Set<String> names() {
    return kernels.keySet();
  }

  @Override
  public void close() {
    RuntimeException first = null;
    for (MonteCarlo<EquityMarket> mc : kernels.values()) {
      try {
        mc.close();
      } catch (RuntimeException e) {
        if (first == null) {
          first = e;
        }
      }
    }
    if (first != null) {
      throw first;
    }
  }

  /** Configures a set of metrics sharing market, step count, precision and engine. */
  public static final class Builder {
    private final EquityMarket market;
    private final Map<String, Product<EquityMarket>> metrics = new LinkedHashMap<>();
    private int steps = 252;
    private boolean fp32;
    private String engine;

    private Builder(EquityMarket market) {
      this.market = market;
    }

    public Builder steps(int steps) {
      this.steps = steps;
      return this;
    }

    public Builder fp32() {
      this.fp32 = true;
      return this;
    }

    public Builder on(String engine) {
      this.engine = engine;
      return this;
    }

    public Builder metric(String name, Product<EquityMarket> product) {
      if (metrics.putIfAbsent(name, product) != null) {
        throw new IllegalArgumentException("duplicate metric name: " + name);
      }
      return this;
    }

    public MultiMetric build() {
      if (metrics.isEmpty()) {
        throw new IllegalStateException("no metrics declared");
      }
      Map<String, MonteCarlo<EquityMarket>> built = new LinkedHashMap<>();
      try {
        metrics.forEach((name, product) -> {
          MonteCarlo.Builder<EquityMarket> b = MonteCarlo.of(product).market(market).steps(steps).greeks();
          b = fp32 ? b.fp32() : b.fp64();
          b = engine == null ? b.fastest() : b.on(engine);
          built.put(name, b.build());
        });
      } catch (RuntimeException e) {
        built.values().forEach(MonteCarlo::close);
        throw e;
      }
      return new MultiMetric(built);
    }
  }
}
