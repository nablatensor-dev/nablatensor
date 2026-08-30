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

import java.util.List;

/**
 * Combines the per-chunk {@link AadResult}s of a segmented replay into one
 * result over the whole path range: a path-count-weighted mean for every output
 * value and every gradient component, and a pooled Monte-Carlo standard error
 * reconstructed from each chunk's {@code (mean, stderr, paths)}.
 *
 * <p>If any chunk reports {@code NaN} for an output's standard error (the engine
 * does not estimate it) the combined standard error for that output is
 * {@code NaN}.
 */
final class AadResultAccumulator {

  private final List<String> outputNames;
  private final List<String> inputNames;
  private final double[] valueSum;       // sum over paths of the per-chunk mean * chunkPaths
  private final double[] squareSum;      // reconstructed sum over paths of value^2
  private final boolean[] stderrKnown;
  private final double[][] gradientSum;  // [output][input] sum over paths
  private double seconds;

  AadResultAccumulator(AadResult shape) {
    this.outputNames = shape.outputNames();
    this.inputNames = shape.inputNames();
    int no = outputNames.size();
    int ni = inputNames.size();
    this.valueSum = new double[no];
    this.squareSum = new double[no];
    this.stderrKnown = new boolean[no];
    this.gradientSum = new double[no][ni];
    java.util.Arrays.fill(stderrKnown, true);
  }

  void add(AadResult partial, long chunkPaths, long totalPaths) {
    seconds += partial.seconds();
    for (int o = 0; o < outputNames.size(); o++) {
      String out = outputNames.get(o);
      double mean = partial.value(out);
      valueSum[o] += mean * chunkPaths;

      double stderr = partial.standardError(out);
      if (Double.isNaN(stderr)) {
        stderrKnown[o] = false;
      } else if (stderrKnown[o]) {
        // stderr = sd / sqrt(n)  ->  sd^2 = stderr^2 * n ;  sum(x^2) = (n-1) sd^2 + n mean^2
        double variance = stderr * stderr * chunkPaths;
        double denom = chunkPaths > 1 ? (chunkPaths - 1) : chunkPaths;
        squareSum[o] += denom * variance + chunkPaths * mean * mean;
      }

      double[] row = gradientSum[o];
      for (int i = 0; i < inputNames.size(); i++) {
        row[i] += partial.gradient(out, inputNames.get(i)) * chunkPaths;
      }
    }
  }

  AadResult result(long totalPaths) {
    int no = outputNames.size();
    double[] values = new double[no];
    double[] stderrs = new double[no];
    double[][] gradients = new double[no][];
    for (int o = 0; o < no; o++) {
      values[o] = valueSum[o] / totalPaths;
      if (stderrKnown[o] && totalPaths > 1) {
        double variance = (squareSum[o] - totalPaths * values[o] * values[o]) / (totalPaths - 1);
        stderrs[o] = Math.sqrt(Math.max(0.0, variance) / totalPaths);
      } else {
        stderrs[o] = Double.NaN;
      }
      double[] row = gradientSum[o].clone();
      for (int i = 0; i < row.length; i++) {
        row[i] /= totalPaths;
      }
      gradients[o] = row;
    }
    return AadResult.of(outputNames, values, stderrs, gradients, inputNames, totalPaths, seconds);
  }
}
