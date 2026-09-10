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
package com.nablatensor.engine.cpu;

import com.nablatensor.engine.AadOp;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.AadTotals;
import com.nablatensor.engine.FlatTape;
import com.nablatensor.engine.HostAadExecutable;

import java.util.Arrays;

/**
 * Replay of a tape one scenario at a time on the JVM.
 *
 * <p>Unlike the CUDA engine there is no code generation: the tape is walked
 * node by node, forward then backward, which costs an interpreter dispatch per
 * node but needs no compiler and runs anywhere. Scenarios are independent, so
 * the only parallelism is the path range each worker is handed by
 * {@link HostAadExecutable}.
 *
 * <p>Both precisions run through this one sweep: the arithmetic is evaluated in
 * {@code double} and, in {@code FLOAT32} mode, every node value and every
 * adjoint is rounded to {@code float} as it is stored. For the forward sweep
 * that is not an approximation of single precision but exactly it —
 * {@code double} carries more than twice the significand of {@code float},
 * which is the condition under which rounding a {@code double} sum, difference,
 * product, quotient or square root to {@code float} yields the bits the
 * {@code float} operation would have produced, and the generated fp32 kernels
 * evaluate the transcendentals in {@code double} and narrow them too. The
 * reverse sweep holds its adjoints in {@code float} the same way, but evaluates
 * each local partial derivative in {@code double} before rounding it, so it is
 * if anything slightly cleaner than a kernel that is single precision
 * throughout. Either way the working precision is the accelerator's, which is
 * what makes this usable as the reference for an fp32 run and not only an fp64
 * one.
 */
final class ScalarReplay extends HostAadExecutable {

  private final FlatTape flat;
  private final boolean f32;
  private final boolean adjoints;

  ScalarReplay(AadTape tape, AadOptions options) {
    super(tape, options, "aad-cpu");
    this.flat = new FlatTape(tape);
    this.f32 = options.precision() == AadOptions.Precision.FLOAT32;
    this.adjoints = options.adjoints();
  }

  @Override
  public String engineName() {
    return "cpu";
  }

  @Override
  public AadResult replay(long paths, long pathOffset, long seed) {
    long start = beginReplay(paths);
    AadTotals total = runRanges(paths, pathOffset, 1L, (from, count) -> run(from, count, seed));
    return total.toResult(tape, paths, (System.nanoTime() - start) / 1e9);
  }

  private AadTotals run(long pathFrom, long count, long seed) {
    final AadOp[] ops = flat.op;
    final int[] argA = flat.argA;
    final int[] argB = flat.argB;
    final double[] constants = flat.constant;
    final boolean[] active = flat.active;
    final int[] inputNodes = flat.inputNode;
    final int[] outputNodes = flat.outputNode;
    final int n = ops.length;
    final int nOut = outputNodes.length;
    final double[] v = new double[n];
    final double[] d = new double[n];
    final double[] in = inputs;
    final AadTotals acc = newTotals();

    final Philox[] rng = new Philox[flat.randStreams];
    for (long path = pathFrom; path < pathFrom + count; path++) {
      for (int s = 0; s < rng.length; s++) {
        rng[s] = new Philox(path, seed, s);
      }
      for (int i = 0; i < n; i++) {
        int a = argA[i];
        int b = argB[i];
        double value = switch (ops[i]) {
          case CONST -> constants[i];
          case INPUT -> in[a];
          case RANDN -> rng[b].normal();
          case RANDU -> rng[b].uniform();
          case ADD -> v[a] + v[b];
          case SUB -> v[a] - v[b];
          case MUL -> v[a] * v[b];
          case DIV -> v[a] / v[b];
          case NEG -> -v[a];
          case EXP -> Math.exp(v[a]);
          case LOG -> Math.log(v[a]);
          case SQRT -> Math.sqrt(v[a]);
          case ABS -> Math.abs(v[a]);
          case MAX -> Math.max(v[a], v[b]);
          case MIN -> Math.min(v[a], v[b]);
        };
        v[i] = f32 ? (float) value : value;
      }
      for (int o = 0; o < nOut; o++) {
        double y = v[outputNodes[o]];
        acc.value[o] += y;
        acc.sumsq[o] += y * y;
      }

      if (!adjoints) {
        continue;
      }
      for (int o = 0; o < nOut; o++) {
        Arrays.fill(d, 0.0);
        d[outputNodes[o]] = 1.0;
        for (int i = n - 1; i >= 0; i--) {
          double adjoint = d[i];
          if (adjoint == 0.0 || !active[i]) {
            continue;
          }
          int a = argA[i];
          int b = argB[i];
          switch (ops[i]) {
            case CONST, INPUT, RANDN, RANDU -> {
            }
            case ADD -> {
              add(d, a, adjoint);
              add(d, b, adjoint);
            }
            case SUB -> {
              add(d, a, adjoint);
              add(d, b, -adjoint);
            }
            case MUL -> {
              add(d, a, adjoint * v[b]);
              add(d, b, adjoint * v[a]);
            }
            case DIV -> {
              add(d, a, adjoint / v[b]);
              add(d, b, -adjoint * v[i] / v[b]);
            }
            case NEG -> add(d, a, -adjoint);
            case EXP -> add(d, a, adjoint * v[i]);
            case LOG -> add(d, a, adjoint / v[a]);
            case SQRT -> add(d, a, adjoint * 0.5 / v[i]);
            case ABS -> add(d, a, v[a] < 0.0 ? -adjoint : adjoint);
            case MAX -> add(d, v[a] >= v[b] ? a : b, adjoint);
            case MIN -> add(d, v[a] <= v[b] ? a : b, adjoint);
          }
        }
        double[] row = acc.gradient[o];
        for (int j = 0; j < row.length; j++) {
          row[j] += d[inputNodes[j]];
        }
      }
    }
    return acc;
  }

  /** Adjoint accumulation, rounded back to the working precision in fp32 mode. */
  private void add(double[] d, int node, double contribution) {
    if (f32) {
      d[node] = (float) (d[node] + (float) contribution);
    } else {
      d[node] += contribution;
    }
  }
}
