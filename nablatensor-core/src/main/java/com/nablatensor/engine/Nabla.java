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
package com.nablatensor.engine;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.ToDoubleFunction;

/**
 * The client-facing entry point: record a valuation once, then value it under
 * as many market states and scenario counts as you like.
 *
 * <pre>{@code
 * try (Nabla.Pricer pricer = Nabla.model(rec -> {
 *         SDouble spot = rec.input("S0", 100.0);
 *         SDouble vol  = rec.input("sigma", 0.2);
 *         ...
 *         rec.output(payoff);
 *       })
 *       .fp32()
 *       .greeks()
 *       .fastest()
 *       .build()) {
 *
 *   Nabla.Valuation v = pricer.value()
 *       .with("S0", 110.0)
 *       .scenarios(10_000_000)
 *       .run();
 *
 *   System.out.println(v.price() + " delta " + v.greek("S0"));
 * }
 * }</pre>
 *
 * <p>The split between {@link Model} and {@link Pricer} is the split between
 * what is expensive and what is not. Building a pricer records the tape and
 * generates a kernel for it; valuing costs a launch. Everything you pass to
 * {@link Request#with} is a kernel argument, so moving the market never rebuilds
 * anything — which is why a pricer is worth holding onto rather than creating
 * per valuation.
 */
public final class Nabla {

  private Nabla() {
  }

  /**
   * The fluent precision / adjoint / engine / tuning knobs shared by every model
   * builder. Implemented once here; a concrete builder only supplies the backing
   * {@link #options()} / {@link #self()} accessors, so a new knob is a single
   * {@code default} method rather than one method per builder.
   *
   * @param <SELF> the concrete builder type, returned for chaining
   */
  public interface ModelConfig<SELF extends ModelConfig<SELF>> {

    /** The options accumulated so far. */
    AadOptions options();

    /** Replaces the accumulated options. */
    void options(AadOptions options);

    /** Pins the engine by name, or {@code null} for automatic selection. */
    void engineName(String engineName);

    /** This builder, for chaining from the shared {@code default} methods. */
    SELF self();

    /** The recorded tape this builder will compile. */
    AadTape tape();

    /** Single precision: the throughput choice, and the default. */
    default SELF fp32() {
      options(options().withPrecision(AadOptions.Precision.FLOAT32));
      return self();
    }

    /** Double precision: slower everywhere, and what to check fp32 against. */
    default SELF fp64() {
      options(options().withPrecision(AadOptions.Precision.FLOAT64));
      return self();
    }

    /** Value and every first-order sensitivity, for barely more than the value. */
    default SELF greeks() {
      options(options().withAdjoints(true));
      return self();
    }

    /** Value only, skipping the adjoint sweep entirely. */
    default SELF priceOnly() {
      options(options().withAdjoints(false));
      return self();
    }

    /** Worker threads for CPU engines; ignored by the CUDA ones. */
    default SELF threads(int count) {
      options(options().withThreads(count));
      return self();
    }

    /** Pins the engine by name: {@code cuda}, {@code cuda-interp}, {@code simd}, {@code cpu}. */
    default SELF on(String engineName) {
      engineName(engineName);
      return self();
    }

    /** Highest-priority engine this machine can actually run. */
    default SELF fastest() {
      engineName(null);
      return self();
    }

    /**
     * Skips per-tape code generation, so the first valuation starts immediately.
     * Worth it when the model changes nearly as often as it is valued.
     */
    default SELF noCompile() {
      engineName("cuda-interp");
      return self();
    }

    /**
     * Enables the given {@code cpu-jit} code-generation optimizations. Every
     * optimization is off by default; other engines ignore this.
     */
    default SELF jit(JitOptimizations optimizations) {
      options(options().withJit(optimizations));
      return self();
    }

    /** Enables a bundle of {@code cpu-jit} optimizations by level. */
    default SELF jit(JitOptimizations.Level level) {
      return jit(JitOptimizations.level(level));
    }

    /** Enables the given {@code cpu-jit} optimization categories. */
    default SELF jit(JitOptimizations.Category... categories) {
      return jit(JitOptimizations.of(categories));
    }

    /**
     * Sets one free-form tuning option for the named engine. {@link #jit} stays
     * the typed surface for {@code cpu-jit}; this is the generic path for other
     * engines' knobs.
     */
    default SELF tune(String engine, String key, String value) {
      options(options().withEngineOption(engine, key, value));
      return self();
    }
  }

  /** Records a valuation written against {@link SDouble} into a reusable model. */
  public static Model model(Consumer<AadRecorder> valuation) {
    long start = System.nanoTime();
    AadTape tape = AadRecorder.record(valuation);
    return new Model(tape, (System.nanoTime() - start) / 1e9);
  }

  /**
   * Records a valuation whose inputs are the components of a market record.
   *
   * <p>The record supplies both the input names and their initial values, so no
   * risk factor is ever named with a string: the valuation reads inputs through
   * accessor references, and {@link TypedValuation#greeks()} hands back the
   * gradient as a record of the same type.
   *
   * <pre>{@code
   * record EquityMarket(double spot, double vol, double rate) {}
   *
   * var market = new EquityMarket(100.0, 0.28, 0.03);
   * try (var pricer = Nabla.model(market, (rec, in) -> {
   *         SDouble spot = in.of(EquityMarket::spot);
   *         ...
   *         rec.output(payoff);
   *       }).fp32().greeks().fastest().build()) {
   *
   *   EquityMarket greeks = pricer.value().with(market).run().greeks();
   *   hedge(greeks.spot());          // delta, checked by the compiler
   * }
   * }</pre>
   */
  public static <M extends Record> TypedModel<M> model(
      M defaults, BiConsumer<AadRecorder, Inputs<M>> valuation) {
    MarketShape<M> shape = MarketShape.of(defaults);
    Model model = model(rec -> valuation.accept(rec, new Inputs<>(rec, shape, defaults)));
    return new TypedModel<>(model, shape);
  }

  /** The market a valuation reads from: one {@link SDouble} per record component. */
  public static final class Inputs<M extends Record> {

    private final MarketShape<M> shape;
    private final SDouble[] inputs;

    private Inputs(AadRecorder recorder, MarketShape<M> shape, M defaults) {
      this.shape = shape;
      this.inputs = new SDouble[shape.size()];
      for (int i = 0; i < inputs.length; i++) {
        inputs[i] = recorder.input(shape.name(i), shape.value(defaults, i));
      }
    }

    /** The input behind one component, named by its accessor rather than a string. */
    public SDouble of(ToDoubleFunction<M> component) {
      return inputs[shape.indexOf(component)];
    }
  }

  /** A recorded valuation over a market record, not yet bound to a device. */
  public static final class TypedModel<M extends Record> implements ModelConfig<TypedModel<M>> {

    private final Model model;
    private final MarketShape<M> shape;

    private TypedModel(Model model, MarketShape<M> shape) {
      this.model = model;
      this.shape = shape;
    }

    @Override
    public AadOptions options() {
      return model.options();
    }

    @Override
    public void options(AadOptions options) {
      model.options(options);
    }

    @Override
    public void engineName(String engineName) {
      model.engineName(engineName);
    }

    @Override
    public TypedModel<M> self() {
      return this;
    }

    @Override
    public AadTape tape() {
      return model.tape();
    }

    public TypedPricer<M> build() {
      return new TypedPricer<>(model.build(), shape);
    }
  }

  /** A built kernel over a market record. */
  public static final class TypedPricer<M extends Record> implements AutoCloseable {

    private final Pricer pricer;
    private final MarketShape<M> shape;

    private TypedPricer(Pricer pricer, MarketShape<M> shape) {
      this.pricer = pricer;
      this.shape = shape;
    }

    public TypedRequest<M> value() {
      return new TypedRequest<>(pricer.value(), shape);
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

    @Override
    public void close() {
      pricer.close();
    }
  }

  /** One valuation, under one market state. */
  public static final class TypedRequest<M extends Record> {

    private final Request request;
    private final MarketShape<M> shape;

    private TypedRequest(Request request, MarketShape<M> shape) {
      this.request = request;
      this.shape = shape;
    }

    /** Sets every risk factor at once. */
    public TypedRequest<M> with(M market) {
      for (int i = 0; i < shape.size(); i++) {
        request.with(shape.name(i), shape.value(market, i));
      }
      return this;
    }

    /** Moves one risk factor, leaving the rest where they were. */
    public TypedRequest<M> with(ToDoubleFunction<M> component, double value) {
      request.with(shape.name(shape.indexOf(component)), value);
      return this;
    }

    public TypedRequest<M> scenarios(long count) {
      request.scenarios(count);
      return this;
    }

    public TypedRequest<M> seed(long value) {
      request.seed(value);
      return this;
    }

    public TypedValuation<M> run() {
      return new TypedValuation<>(request.run(), shape);
    }
  }

  /** What came back, with the gradient shaped like the market. */
  public record TypedValuation<M extends Record>(Valuation valuation, MarketShape<M> shape) {

    public double price() {
      return valuation.price();
    }

    /** Every sensitivity, as a record of the same type as the market. */
    public M greeks() {
      double[] gradient = new double[shape.size()];
      for (int i = 0; i < gradient.length; i++) {
        gradient[i] = valuation.greek(shape.name(i));
      }
      return shape.build(gradient);
    }

    /** One sensitivity, named by its accessor. */
    public double greek(ToDoubleFunction<M> component) {
      return valuation.greek(shape.name(shape.indexOf(component)));
    }

    /** Every sensitivity of a named output, as a record of the same type as the market. */
    public M greeks(String output) {
      double[] gradient = new double[shape.size()];
      for (int i = 0; i < gradient.length; i++) {
        gradient[i] = valuation.greek(output, shape.name(i));
      }
      return shape.build(gradient);
    }

    /** One sensitivity of a named output, named by its accessor. */
    public double greek(String output, ToDoubleFunction<M> component) {
      return valuation.greek(output, shape.name(shape.indexOf(component)));
    }

    /** Monte-Carlo standard error of {@link #price()}, or {@code NaN} if not estimated. */
    public double standardError() {
      return valuation.standardError();
    }

    public long scenarios() {
      return valuation.scenarios();
    }

    public double seconds() {
      return valuation.seconds();
    }

    public double scenariosPerSecond() {
      return valuation.scenariosPerSecond();
    }
  }


  /** A recorded valuation, not yet bound to a device or a precision. */
  public static final class Model implements ModelConfig<Model> {

    private final AadTape tape;
    private final double recordSeconds;
    private AadOptions options = AadOptions.defaults();
    private String engine;

    private Model(AadTape tape, double recordSeconds) {
      this.tape = tape;
      this.recordSeconds = recordSeconds;
    }

    @Override
    public AadOptions options() {
      return options;
    }

    @Override
    public void options(AadOptions options) {
      this.options = options;
    }

    @Override
    public void engineName(String engineName) {
      this.engine = engineName;
    }

    @Override
    public Model self() {
      return this;
    }

    @Override
    public AadTape tape() {
      return tape;
    }

    public Pricer build() {
      long start = System.nanoTime();
      AadEngine selected = engine == null
          ? AadEngines.select(options)
          : AadEngines.require(engine, options);
      AadExecutable executable = selected.compile(tape, options);
      return new Pricer(executable, recordSeconds, (System.nanoTime() - start) / 1e9);
    }
  }

  /** A built kernel. Hold it; valuing through it is cheap, building it is not. */
  public static final class Pricer implements AutoCloseable {

    private final AadExecutable executable;
    private final double recordSeconds;
    private final double buildSeconds;

    private Pricer(AadExecutable executable, double recordSeconds, double buildSeconds) {
      this.executable = executable;
      this.recordSeconds = recordSeconds;
      this.buildSeconds = buildSeconds;
    }

    public Request value() {
      return new Request(executable);
    }

    public String engine() {
      return executable.engineName();
    }

    public int nodes() {
      return executable.tape().size();
    }

    public double recordSeconds() {
      return recordSeconds;
    }

    /** Wall clock spent turning the tape into something runnable. */
    public double buildSeconds() {
      return buildSeconds;
    }

    @Override
    public void close() {
      executable.close();
    }
  }

  /** One valuation, under one market state, over one batch of scenarios. */
  public static final class Request {

    private static final long DEFAULT_SEED = 0x9E3779B97F4A7C15L;

    private final AadExecutable executable;
    private final Map<String, Double> market = new LinkedHashMap<>();
    private long scenarios = 1_000_000L;
    private long seed = DEFAULT_SEED;

    private Request(AadExecutable executable) {
      this.executable = executable;
    }

    /** Sets one market input. Kernel argument, so this never rebuilds anything. */
    public Request with(String input, double value) {
      market.put(input, value);
      return this;
    }

    public Request scenarios(long count) {
      this.scenarios = count;
      return this;
    }

    /** Fixes the scenario stream, so the same seed reproduces the same numbers. */
    public Request seed(long value) {
      this.seed = value;
      return this;
    }

    public Valuation run() {
      market.forEach(executable::setInput);
      return new Valuation(executable.replaySafe(scenarios, seed));
    }
  }

  /** What came back. */
  public record Valuation(AadResult result) {

    public double price() {
      return result.value();
    }

    /** Mean of a named output, for a tape recorded with several {@code rec.output(name, ...)}. */
    public double price(String output) {
      return result.value(output);
    }

    /** Monte-Carlo standard error of {@link #price()}, or {@code NaN} if the engine does not estimate it. */
    public double standardError() {
      return result.standardError();
    }

    /** Monte-Carlo standard error of a named output's mean. */
    public double standardError(String output) {
      return result.standardError(output);
    }

    /** The recorded output names, in recording order. */
    public java.util.List<String> outputNames() {
      return result.outputNames();
    }

    /** Sensitivity of the price to a named input. */
    public double greek(String input) {
      return result.gradient(input);
    }

    /** Sensitivity of a named output to a named input. */
    public double greek(String output, String input) {
      return result.gradient(output, input);
    }

    /** Every sensitivity, in the order the inputs were declared. */
    public Map<String, Double> greeks() {
      Map<String, Double> all = new LinkedHashMap<>();
      for (String name : result.inputNames()) {
        all.put(name, result.gradient(name));
      }
      return all;
    }

    /** Every sensitivity of a named output. */
    public Map<String, Double> greeks(String output) {
      return result.gradients(output);
    }

    public long scenarios() {
      return result.paths();
    }

    public double seconds() {
      return result.seconds();
    }

    public double scenariosPerSecond() {
      return result.pathsPerSecond();
    }
  }
}
