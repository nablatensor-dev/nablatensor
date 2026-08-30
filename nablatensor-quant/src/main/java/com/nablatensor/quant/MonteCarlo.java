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

import com.nablatensor.engine.JitOptimizations;
import com.nablatensor.engine.Nabla;

/**
 * Records a {@link Product} once against a market record {@code M} and replays it
 * for price and every Greek from one adjoint sweep. {@code M} is
 * {@link EquityMarket} for the built-in catalogue; any {@code double}-only record
 * works.
 *
 * <pre>{@code
 * try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.asianCall())
 *         .market(EquityMarket.atmOneYear())
 *         .steps(252)                       // or .timeGrid(TimeGrid.of(t1, t2, ...))
 *         .greeks()
 *         .fastest()
 *         .build()) {
 *
 *   Nabla.TypedValuation<EquityMarket> p = mc.run(1_000_000, 42L);
 *   System.out.println(p.price() + " delta " + p.greek(EquityMarket::spot));
 *
 *   // Seam 2: move the market on the compiled kernel, no re-record.
 *   var bumped = mc.run(mc.market().withSpot(101.0), 1_000_000, 42L);
 * }
 * }</pre>
 *
 * <p>Building records the tape and generates the kernel; running is a launch.
 * Hold the {@code MonteCarlo}, re-run it under as many markets and seeds as you
 * like.
 *
 * @param <M> the market record the payoff reads its differentiable inputs from
 */
public final class MonteCarlo<M extends Record> implements AutoCloseable {

  private final Nabla.TypedPricer<M> pricer;
  private final M market;
  private final boolean greeks;

  private MonteCarlo(Nabla.TypedPricer<M> pricer, M market, boolean greeks) {
    this.pricer = pricer;
    this.market = market;
    this.greeks = greeks;
  }

  public static <M extends Record> Builder<M> of(Product<M> product) {
    return new Builder<>(product);
  }

  /** The market the tape was recorded against; the default for {@link #run(long, long)}. */
  public M market() {
    return market;
  }

  /** Whether this kernel was built with the adjoint sweep. */
  public boolean hasGreeks() {
    return greeks;
  }

  public String engine() {
    return pricer.engine();
  }

  public int nodes() {
    return pricer.nodes();
  }

  public double recordSeconds() {
    return pricer.recordSeconds();
  }

  public double buildSeconds() {
    return pricer.buildSeconds();
  }

  /** Replays under the recorded market. */
  public Nabla.TypedValuation<M> run(long scenarios, long seed) {
    return run(market, scenarios, seed);
  }

  /** Replays under an arbitrary market state — a kernel argument, so nothing rebuilds. */
  public Nabla.TypedValuation<M> run(M state, long scenarios, long seed) {
    return pricer.value().with(state).scenarios(scenarios).seed(seed).run();
  }

  @Override
  public void close() {
    pricer.close();
  }

  /** Fluent configuration for a {@link MonteCarlo}. Mirrors the engine's own model builder. */
  public static final class Builder<M extends Record> {

    private final Product<M> product;
    private M market;
    private TimeGrid grid = TimeGrid.uniform(252);
    private boolean greeks = true;
    private boolean fp32;
    private int threads;
    private String engine;
    private JitOptimizations.Level jit;

    private Builder(Product<M> product) {
      this.product = product;
    }

    /** The market record to record the tape against; required. */
    public Builder<M> market(M market) {
      this.market = market;
      return this;
    }

    /** {@code n} equally spaced time steps per path. */
    public Builder<M> steps(int steps) {
      this.grid = TimeGrid.uniform(steps);
      return this;
    }

    /** An explicit (possibly non-uniform) simulation schedule. */
    public Builder<M> timeGrid(TimeGrid grid) {
      this.grid = grid;
      return this;
    }

    /** Value and every first-order sensitivity (the default). */
    public Builder<M> greeks() {
      this.greeks = true;
      return this;
    }

    /** Value only; skip the adjoint sweep. */
    public Builder<M> priceOnly() {
      this.greeks = false;
      return this;
    }

    public Builder<M> fp32() {
      this.fp32 = true;
      return this;
    }

    public Builder<M> fp64() {
      this.fp32 = false;
      return this;
    }

    public Builder<M> threads(int threads) {
      this.threads = threads;
      return this;
    }

    /** Pin the backend by name: {@code cpu}, {@code cpu-jit}, {@code simd}. */
    public Builder<M> on(String engine) {
      this.engine = engine;
      return this;
    }

    /** Highest-priority backend this machine can run (the default). */
    public Builder<M> fastest() {
      this.engine = null;
      return this;
    }

    /** Enable a bundle of {@code cpu-jit} code-generation optimizations; off by default. */
    public Builder<M> jit(JitOptimizations.Level level) {
      this.jit = level;
      return this;
    }

    public MonteCarlo<M> build() {
      if (market == null) {
        throw new IllegalStateException("MonteCarlo.Builder needs .market(...) before .build()");
      }
      if (market instanceof EquityMarket em) {
        em.validated();   // reject an unsimulatable equity market; returns the same instance
      }
      final Product<M> p = product;
      final TimeGrid g = grid;
      Nabla.TypedModel<M> model = Nabla.model(market, (rec, in) -> p.record(rec, in, g));
      model = fp32 ? model.fp32() : model.fp64();
      model = greeks ? model.greeks() : model.priceOnly();
      if (threads > 0) {
        model = model.threads(threads);
      }
      if (jit != null) {
        model = model.jit(jit);
      }
      model = engine == null ? model.fastest() : model.on(engine);
      return new MonteCarlo<>(model.build(), market, greeks);
    }
  }
}
