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

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.Nabla;
import com.nablatensor.engine.SDouble;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Price several named risk measures from <strong>one recorded tape and one
 * compiled kernel</strong>: the tape carries one {@code rec.output(name, ...)}
 * per measure, and a single replay runs the forward sweep once and one reverse
 * sweep per measure, returning every value, its Monte-Carlo standard error, and
 * its full input gradient — the {@code N × M} Jacobian — all from the same
 * Philox paths.
 *
 * <pre>{@code
 * try (MultiOutput mo = MultiOutput.of(rec -> {
 *         SDouble s0 = rec.input("S0", 100), k = rec.input("K", 100),
 *                 vol = rec.input("sigma", 0.2), r = rec.input("r", 0.03);
 *         SDouble sT = s0.mul(r.sub(vol.mul(vol).mul(0.5)).add(vol.mul(rec.randn())).exp());
 *         return Map.of(
 *             "call",    sT.sub(k).max(0.0).mul(r.neg().exp()),
 *             "digital", com.nablatensor.ops.Smooth.gt(rec, sT, k, 1.0).mul(r.neg().exp()));
 *       }).on("cpu-jit").build()) {
 *
 *   MultiOutput.Result res = mo.run(1_000_000, 42L);
 *   res.value("digital");                 // digital price
 *   res.gradient("call").get("sigma");    // call vega
 * }
 * }</pre>
 */
public final class MultiOutput implements AutoCloseable {

  /**
   * Records the shared valuation and returns the named result measures. Use a
   * {@link LinkedHashMap} (not {@link Map#of}) if a stable output order matters.
   */
  @FunctionalInterface
  public interface Measures {
    Map<String, SDouble> record(AadRecorder rec);
  }

  private final Nabla.Pricer pricer;
  private final List<String> outputs;
  private final List<String> inputs;

  private MultiOutput(Nabla.Pricer pricer, List<String> outputs, List<String> inputs) {
    this.pricer = pricer;
    this.outputs = outputs;
    this.inputs = inputs;
  }

  public static Builder of(Measures measures) {
    return new Builder(measures);
  }

  public List<String> outputNames() {
    return List.copyOf(outputs);
  }

  public String engine() {
    return pricer.engine();
  }

  public int nodes() {
    return pricer.nodes();
  }

  /** Values + full Jacobian, using the recorded input values. */
  public Result run(long scenarios, long seed) {
    return run(Map.of(), scenarios, seed);
  }

  /** Values + full Jacobian, with named inputs overridden. */
  public Result run(Map<String, Double> marketOverrides, long scenarios, long seed) {
    Nabla.Request req = pricer.value();
    marketOverrides.forEach(req::with);
    Nabla.Valuation v = req.scenarios(scenarios).seed(seed).run();

    Map<String, Double> values = new LinkedHashMap<>();
    Map<String, Double> stderr = new LinkedHashMap<>();
    Map<String, Map<String, Double>> gradients = new LinkedHashMap<>();
    for (String o : outputs) {
      values.put(o, v.price(o));
      stderr.put(o, v.standardError(o));
      gradients.put(o, v.greeks(o));
    }
    return new Result(values, stderr, gradients, scenarios, outputs);
  }

  @Override
  public void close() {
    pricer.close();
  }

  /** The N values, their standard errors, and the N×M Jacobian from one {@link #run}. */
  public record Result(Map<String, Double> values, Map<String, Double> standardErrors,
                       Map<String, Map<String, Double>> gradients,
                       long scenarios, List<String> outputOrder) {

    public double value(String output) {
      return require(values, output);
    }

    public double standardError(String output) {
      return require(standardErrors, output);
    }

    /** {@code d(output) / d(input)} for every recorded input. */
    public Map<String, Double> gradient(String output) {
      Map<String, Double> g = gradients.get(output);
      if (g == null) {
        throw new IllegalArgumentException("unknown output '" + output + "'; have " + values.keySet());
      }
      return g;
    }

    public double sensitivity(String output, String input) {
      Double d = gradient(output).get(input);
      if (d == null) {
        throw new IllegalArgumentException("unknown input '" + input + "'");
      }
      return d;
    }

    private static double require(Map<String, Double> m, String output) {
      Double v = m.get(output);
      if (v == null) {
        throw new IllegalArgumentException("unknown output '" + output + "'; have " + m.keySet());
      }
      return v;
    }
  }

  /** Fluent configuration. */
  public static final class Builder {
    private final Measures measures;
    private boolean fp32;
    private int threads;
    private String engine = "cpu-jit";

    private Builder(Measures measures) {
      this.measures = measures;
    }

    public Builder fp32() {
      this.fp32 = true;
      return this;
    }

    public Builder threads(int n) {
      this.threads = n;
      return this;
    }

    public Builder on(String engine) {
      this.engine = engine;
      return this;
    }

    public MultiOutput build() {
      List<String> order = new ArrayList<>();
      Nabla.Model model = Nabla.model(rec -> {
        Map<String, SDouble> outs = measures.record(rec);
        if (outs.isEmpty()) {
          throw new IllegalStateException("no measures recorded");
        }
        outs.forEach((name, value) -> {
          order.add(name);
          rec.output(name, value);
        });
      });

      List<String> inputs = List.copyOf(model.tape().inputNames());
      model = fp32 ? model.fp32() : model.fp64();
      model = model.greeks();
      if (threads > 0) {
        model = model.threads(threads);
      }
      model = model.on(engine);
      return new MultiOutput(model.build(), List.copyOf(order), inputs);
    }
  }
}
