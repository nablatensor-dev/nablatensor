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

import com.nablatensor.annotation.Internal;

/**
 * Per-worker running sums for a host replay: for each recorded output the sum
 * of its value over scenarios, the sum of its square, and the sum of its
 * adjoint with respect to each differentiable input.
 *
 * <p>One instance per worker thread, combined with {@link #add} once the
 * workers are done and turned into means, a Monte-Carlo standard error and mean
 * gradients by {@link #toResult}. Keeping the sums here — rather than a copy of
 * the same three arrays inside every engine — is what lets the scalar, the
 * generated-bytecode and the vector sweeps all report a standard error from one
 * piece of code.
 *
 * <p>Public only so the engine modules can use it; not part of the supported
 * API.
 */
@Internal
public final class AadTotals {

  /** Per output: sum over scenarios of the output value. */
  public final double[] value;
  /** Per output: sum over scenarios of the output value squared. */
  public final double[] sumsq;
  /** {@code [output][input]}: sum over scenarios of the adjoint. */
  public final double[][] gradient;

  public AadTotals(int outputs, int inputs) {
    this.value = new double[outputs];
    this.sumsq = new double[outputs];
    this.gradient = new double[outputs][inputs];
  }

  public void add(AadTotals other) {
    for (int o = 0; o < value.length; o++) {
      value[o] += other.value[o];
      sumsq[o] += other.sumsq[o];
      double[] row = gradient[o];
      double[] otherRow = other.gradient[o];
      for (int j = 0; j < row.length; j++) {
        row[j] += otherRow[j];
      }
    }
  }

  /**
   * The sums divided down into a result: mean value, the standard error of that
   * mean, and the mean adjoint per input. The variance uses the raw
   * sum-of-squares form, which is what a per-worker sum can carry cheaply.
   */
  public AadResult toResult(AadTape tape, long paths, double seconds) {
    int outputs = value.length;
    double[] means = new double[outputs];
    double[] stderr = new double[outputs];
    double[][] gradients = new double[outputs][];
    for (int o = 0; o < outputs; o++) {
      means[o] = value[o] / paths;
      if (paths > 1) {
        double variance = (sumsq[o] - paths * means[o] * means[o]) / (paths - 1);
        stderr[o] = Math.sqrt(Math.max(0.0, variance) / paths);
      } else {
        stderr[o] = Double.NaN;
      }
      double[] row = gradient[o].clone();
      for (int j = 0; j < row.length; j++) {
        row[j] /= paths;
      }
      gradients[o] = row;
    }
    return AadResult.of(tape.outputNames(), means, stderr, gradients, tape.inputNames(),
        paths, seconds);
  }
}
