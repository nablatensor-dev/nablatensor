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

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadEngines;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Product;
import com.nablatensor.engine.Nabla;
import java.util.ArrayList;
import java.util.List;

/**
 * Model-validation harness: record one valuation, replay it on every backend
 * this machine can run at an equal seed, and diff each against the scalar CPU
 * oracle; then cross-check the oracle's adjoint gradient against a central bump.
 *
 * <p>The output is a {@link Report} that renders as a plain-text evidence pack —
 * the machine, the seed, the scenario count, and a pass/fail line per backend.
 *
 * <pre>{@code
 * Report r = ModelValidation.of(Products.asianCall())
 *     .market(EquityMarket.atmOneYear()).steps(252)
 *     .scenarios(1_000_000).seed(42L)
 *     .run();
 * System.out.println(r);
 * if (!r.passed()) throw new AssertionError(r.firstFailure());
 * }</pre>
 */
public final class ModelValidation {

  private final Product<EquityMarket> product;
  private EquityMarket market = EquityMarket.atmOneYear();
  private int steps = 252;
  private long scenarios = 1_000_000L;
  private long seed = 0x9E3779B97F4A7C15L;
  private boolean fp32 = false;
  private double tolerance = 1e-6;
  private double bump = 5e-3;

  private ModelValidation(Product<EquityMarket> product) {
    this.product = product;
  }

  public static ModelValidation of(Product<EquityMarket> product) {
    return new ModelValidation(product);
  }

  public ModelValidation market(EquityMarket market) {
    this.market = market;
    return this;
  }

  public ModelValidation steps(int steps) {
    this.steps = steps;
    return this;
  }

  public ModelValidation scenarios(long scenarios) {
    this.scenarios = scenarios;
    return this;
  }

  public ModelValidation seed(long seed) {
    this.seed = seed;
    return this;
  }

  public ModelValidation fp32() {
    this.fp32 = true;
    return this;
  }

  public ModelValidation fp64() {
    this.fp32 = false;
    return this;
  }

  /** Max relative difference a backend may show against the oracle and still pass. */
  public ModelValidation tolerance(double tolerance) {
    this.tolerance = tolerance;
    return this;
  }

  /** Relative bump size for the adjoint cross-check. */
  public ModelValidation bump(double bump) {
    this.bump = bump;
    return this;
  }

  public Report run() {
    Nabla.TypedValuation<EquityMarket> oracle;
    try (MonteCarlo<EquityMarket> mc = build("cpu")) {
      oracle = mc.run(market, scenarios, seed);
    }

    List<EngineComparison> comparisons = new ArrayList<>();
    for (String engine : candidateEngines()) {
      try (MonteCarlo<EquityMarket> mc = build(engine)) {
        Nabla.TypedValuation<EquityMarket> candidate = mc.run(market, scenarios, seed);
        comparisons.add(EngineComparison.of(engine, mc.engine(), oracle, candidate, tolerance));
      } catch (RuntimeException | LinkageError e) {
        comparisons.add(new EngineComparison(engine, "unavailable: " + e, Double.NaN, Double.NaN,
            Double.NaN, Double.NaN, false));
      }
    }

    BumpCrossCheck crossCheck = BumpCrossCheck.run(
        product, market, steps, fp32, scenarios, seed, bump,
        oracle.greeks());

    return new Report(product.label(), market, steps, scenarios, seed, fp32, tolerance,
        machineInfo(), oracle, comparisons, crossCheck);
  }

  private MonteCarlo<EquityMarket> build(String engine) {
    MonteCarlo.Builder<EquityMarket> b = MonteCarlo.of(product).market(market).steps(steps).greeks().on(engine);
    return (fp32 ? b.fp32() : b.fp64()).build();
  }

  private List<String> candidateEngines() {
    AadOptions options = new AadOptions(
        fp32 ? AadOptions.Precision.FLOAT32 : AadOptions.Precision.FLOAT64, true);
    List<String> names = new ArrayList<>();
    for (AadEngine engine : AadEngines.available(options)) {
      if (!engine.name().equals("cpu")) {
        names.add(engine.name());
      }
    }
    return names;
  }

  /** One line describing the JVM and host, for the evidence pack header. */
  public static String machineInfo() {
    Runtime.Version v = Runtime.version();
    return "JDK " + v + " · " + System.getProperty("os.name") + " " + System.getProperty("os.arch")
        + " · " + Runtime.getRuntime().availableProcessors() + " processors";
  }
}
