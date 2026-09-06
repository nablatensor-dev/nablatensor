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
package com.nablatensor.risk;

import java.util.Arrays;

/**
 * A sample of portfolio profit-and-loss — one value per historical day or per
 * revaluation scenario. Positive is a gain; a loss is {@code -pnl}. This is the
 * raw input to {@link ValueAtRisk#historical} and
 * {@link ValueAtRisk#expectedShortfall}, and the realised series a
 * {@link VarBacktest} scores a forecast against.
 */
public record PnlVector(double[] pnl) {

  public PnlVector {
    if (pnl.length == 0) {
      throw new IllegalArgumentException("pnl sample must be non-empty");
    }
    for (double v : pnl) {
      if (!Double.isFinite(v)) {
        throw new IllegalArgumentException("pnl sample contains a non-finite value: " + v);
      }
    }
    pnl = pnl.clone();
  }

  @Override
  public double[] pnl() {
    return pnl.clone();
  }

  public int size() {
    return pnl.length;
  }

  public double mean() {
    double s = 0.0;
    for (double v : pnl) {
      s += v;
    }
    return s / pnl.length;
  }

  public double standardDeviation() {
    double m = mean();
    double s = 0.0;
    for (double v : pnl) {
      double d = v - m;
      s += d * d;
    }
    return Math.sqrt(s / Math.max(1, pnl.length - 1));
  }

  /** Losses ({@code -pnl}) in ascending order. */
  public double[] sortedLosses() {
    double[] loss = new double[pnl.length];
    for (int i = 0; i < pnl.length; i++) {
      loss[i] = -pnl[i];
    }
    Arrays.sort(loss);
    return loss;
  }
}
