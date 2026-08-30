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
import java.util.List;
import java.util.Map;

/**
 * The outcome of replaying a compiled tape over a batch of scenarios: for every
 * recorded output, its mean value, the Monte-Carlo standard error of that mean,
 * and the mean adjoint gradient with respect to each differentiable input.
 *
 * <p>A tape recorded with a single {@code rec.output(...)} has one output named
 * {@code "value"}; the no-argument {@link #value()}, {@link #standardError()},
 * {@link #gradient(String)} and {@link #gradients()} accessors read that
 * <em>primary</em> output (the first one recorded) and are all a single-output
 * caller ever needs. A tape with several {@code rec.output(name, ...)} calls
 * carries one value / stderr / gradient row per name, addressable through the
 * {@code (output, input)} overloads.
 *
 * <p>{@code standardError} is {@link Double#NaN} when the producing engine does
 * not yet estimate it.
 */
public final class AadResult {

  private final String[] outputName;
  private final double[] outputValue;
  private final double[] outputStdErr;
  private final double[][] outputGradient;   // [output][input]
  private final String[] inputName;
  private final Map<String, Integer> inputIndex;
  private final long paths;
  private final double seconds;

  /** Single-output result without a standard-error estimate. */
  public AadResult(double value, double[] gradients, List<String> inputNames,
                   long paths, double seconds) {
    this(value, Double.NaN, gradients, inputNames, paths, seconds);
  }

  /** Single-output result, output named {@code "value"}. */
  public AadResult(double value, double standardError, double[] gradients,
                   List<String> inputNames, long paths, double seconds) {
    this(new String[] {"value"}, new double[] {value}, new double[] {standardError},
        new double[][] {gradients.clone()}, inputNames, paths, seconds);
  }

  private AadResult(String[] outputName, double[] outputValue, double[] outputStdErr,
                    double[][] outputGradient, List<String> inputNames,
                    long paths, double seconds) {
    if (outputName.length == 0) {
      throw new IllegalArgumentException("a result needs at least one output");
    }
    this.outputName = outputName;
    this.outputValue = outputValue;
    this.outputStdErr = outputStdErr;
    this.outputGradient = outputGradient;
    this.inputName = inputNames.toArray(String[]::new);
    Map<String, Integer> index = new LinkedHashMap<>();
    for (int i = 0; i < this.inputName.length; i++) {
      index.put(this.inputName[i], i);
    }
    this.inputIndex = index;
    this.paths = paths;
    this.seconds = seconds;
  }

  /**
   * Multi-output result. {@code values}, {@code standardErrors} and
   * {@code gradients} are indexed by the position of a name in
   * {@code outputNames}; each {@code gradients[o]} is indexed like
   * {@code inputNames}.
   */
  public static AadResult of(List<String> outputNames, double[] values, double[] standardErrors,
                             double[][] gradients, List<String> inputNames,
                             long paths, double seconds) {
    return new AadResult(outputNames.toArray(String[]::new), values.clone(),
        standardErrors.clone(), gradients.clone(), inputNames, paths, seconds);
  }

  // ---- primary output (single-output convenience) -----------------------

  /** Mean value of the primary (first-recorded) output. */
  public double value() {
    return outputValue[0];
  }

  /** Monte-Carlo standard error of {@link #value()}, or {@code NaN} if not estimated. */
  public double standardError() {
    return outputStdErr[0];
  }

  /** Mean adjoint of the primary output with respect to each input, input order. */
  public double[] gradients() {
    return outputGradient[0];
  }

  /** Mean adjoint of the primary output with respect to one named input. */
  public double gradient(String inputName) {
    return outputGradient[0][inputIndexOf(inputName)];
  }

  // ---- per-output (multi-output) --------------------------------------

  /** The recorded output names, in recording order. */
  public List<String> outputNames() {
    return List.of(outputName);
  }

  public int outputCount() {
    return outputName.length;
  }

  public double value(String outputName) {
    return outputValue[outputIndexOf(outputName)];
  }

  public double standardError(String outputName) {
    return outputStdErr[outputIndexOf(outputName)];
  }

  /** {@code d(output) / d(input)}. */
  public double gradient(String outputName, String inputName) {
    return outputGradient[outputIndexOf(outputName)][inputIndexOf(inputName)];
  }

  /** The full gradient row for one output, as an ordered {@code input -> adjoint} map. */
  public Map<String, Double> gradients(String outputName) {
    double[] row = outputGradient[outputIndexOf(outputName)];
    Map<String, Double> out = new LinkedHashMap<>();
    for (int i = 0; i < inputName.length; i++) {
      out.put(inputName[i], row[i]);
    }
    return out;
  }

  // ---- shared ---------------------------------------------------------

  public List<String> inputNames() {
    return List.of(inputName);
  }

  public long paths() {
    return paths;
  }

  public double seconds() {
    return seconds;
  }

  public double pathsPerSecond() {
    return paths / seconds;
  }

  private int outputIndexOf(String name) {
    for (int i = 0; i < outputName.length; i++) {
      if (outputName[i].equals(name)) {
        return i;
      }
    }
    throw new IllegalArgumentException("unknown output: " + name + " (have " + List.of(outputName) + ")");
  }

  private int inputIndexOf(String name) {
    Integer i = inputIndex.get(name);
    if (i == null) {
      throw new IllegalArgumentException("unknown input: " + name);
    }
    return i;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("AadResult(");
    for (int o = 0; o < outputName.length; o++) {
      if (o > 0) {
        sb.append(", ");
      }
      sb.append(outputName[o]).append('=').append(outputValue[o]);
    }
    return sb.append(", paths=").append(paths).append(')').toString();
  }
}
