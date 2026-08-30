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

import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.AbstractAadExecutable;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Double-precision replay of a tape one scenario at a time on the JVM.
 *
 * <p>Unlike the CUDA engine there is no code generation: the tape is walked
 * node by node, forward then backward, which costs an interpreter dispatch per
 * node but needs no compiler and runs anywhere. Scenarios are independent, so
 * the only parallelism is splitting the path range across worker threads.
 */
final class ScalarReplay extends AbstractAadExecutable {

  private final int threads;
  private final ExecutorService pool;

  // Flattened once so the sweeps read plain arrays instead of calling through
  // the tape for every node of every path.
  private final com.nablatensor.engine.AadOp[] ops;
  private final int[] argA;
  private final int[] argB;
  private final double[] constants;
  private final boolean[] active;
  private final int[] inputNodes;
  private final int[] outputNodes;
  private final int streams;

  ScalarReplay(AadTape tape, AadOptions options) {
    super(tape, options);
    this.threads = Math.max(1, options.resolvedThreads());
    this.pool = threads > 1
        ? Executors.newFixedThreadPool(threads, runnable -> {
            Thread thread = new Thread(runnable, "aad-cpu");
            thread.setDaemon(true);
            return thread;
          })
        : null;

    int n = tape.size();
    this.ops = new com.nablatensor.engine.AadOp[n];
    this.argA = new int[n];
    this.argB = new int[n];
    this.constants = new double[n];
    this.active = new boolean[n];
    for (int i = 0; i < n; i++) {
      ops[i] = tape.op(i);
      argA[i] = tape.argA(i);
      argB[i] = tape.argB(i);
      constants[i] = tape.constant(i);
      active[i] = tape.isActive(i);
    }
    this.inputNodes = new int[tape.inputCount()];
    for (int j = 0; j < inputNodes.length; j++) {
      inputNodes[j] = tape.inputNode(j);
    }
    this.outputNodes = new int[tape.outputCount()];
    for (int o = 0; o < outputNodes.length; o++) {
      outputNodes[o] = tape.outputNode(o);
    }
    this.streams = tape.randStreamCount();
  }

  @Override
  public String engineName() {
    return "cpu";
  }

  @Override
  public AadResult replay(long paths, long pathOffset, long seed) {
    checkOpen();
    if (paths <= 0) {
      throw new IllegalArgumentException("paths must be positive");
    }
    long start = System.nanoTime();
    Accumulator total = threads == 1
        ? run(pathOffset, paths, seed)
        : runParallel(paths, pathOffset, seed);
    double seconds = (System.nanoTime() - start) / 1e9;
    return total.toResult(tape, paths, seconds);
  }

  private Accumulator runParallel(long paths, long pathOffset, long seed) {
    List<Callable<Accumulator>> tasks = new ArrayList<>(threads);
    long each = paths / threads;
    long extra = paths % threads;
    long cursor = pathOffset;
    for (int t = 0; t < threads; t++) {
      long count = each + (t < extra ? 1 : 0);
      long from = cursor;
      cursor += count;
      if (count > 0) {
        tasks.add(() -> run(from, count, seed));
      }
    }
    Accumulator total = new Accumulator(outputNodes.length, tape.inputCount());
    try {
      for (Future<Accumulator> future : pool.invokeAll(tasks)) {
        total.add(future.get());
      }
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("replay interrupted", interrupted);
    } catch (java.util.concurrent.ExecutionException failure) {
      Throwable cause = failure.getCause();
      throw cause instanceof RuntimeException runtime ? runtime : new IllegalStateException(cause);
    }
    return total;
  }

  private Accumulator run(long pathFrom, long count, long seed) {
    final int n = ops.length;
    final double[] v = new double[n];
    final double[] d = new double[n];
    final double[] in = inputs;
    final boolean adjoints = options.adjoints();
    final int nOut = outputNodes.length;
    final Accumulator acc = new Accumulator(nOut, inputNodes.length);

    final Philox[] rng = new Philox[streams];
    for (long path = pathFrom; path < pathFrom + count; path++) {
      for (int s = 0; s < streams; s++) {
        rng[s] = new Philox(path, seed, s);
      }
      for (int i = 0; i < n; i++) {
        int a = argA[i];
        int b = argB[i];
        v[i] = switch (ops[i]) {
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
        java.util.Arrays.fill(d, 0.0);
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
              d[a] += adjoint;
              d[b] += adjoint;
            }
            case SUB -> {
              d[a] += adjoint;
              d[b] -= adjoint;
            }
            case MUL -> {
              d[a] += adjoint * v[b];
              d[b] += adjoint * v[a];
            }
            case DIV -> {
              d[a] += adjoint / v[b];
              d[b] -= adjoint * v[i] / v[b];
            }
            case NEG -> d[a] -= adjoint;
            case EXP -> d[a] += adjoint * v[i];
            case LOG -> d[a] += adjoint / v[a];
            case SQRT -> d[a] += adjoint * 0.5 / v[i];
            case ABS -> d[a] += v[a] < 0.0 ? -adjoint : adjoint;
            case MAX -> {
              if (v[a] >= v[b]) d[a] += adjoint; else d[b] += adjoint;
            }
            case MIN -> {
              if (v[a] <= v[b]) d[a] += adjoint; else d[b] += adjoint;
            }
          }
        }
        double[] grow = acc.gradient[o];
        for (int j = 0; j < grow.length; j++) {
          grow[j] += d[inputNodes[j]];
        }
      }
    }
    return acc;
  }

  @Override
  public void close() {
    super.close();
    if (pool != null) {
      pool.shutdownNow();
    }
  }

  private static final class Accumulator {
    final double[] value;              // per output: sum over paths of the output value
    final double[] sumsq;             // per output: sum over paths of value^2
    final double[][] gradient;        // [output][input]: sum over paths of the adjoint

    Accumulator(int outputs, int inputs) {
      this.value = new double[outputs];
      this.sumsq = new double[outputs];
      this.gradient = new double[outputs][inputs];
    }

    void add(Accumulator other) {
      for (int o = 0; o < value.length; o++) {
        value[o] += other.value[o];
        sumsq[o] += other.sumsq[o];
        double[] row = gradient[o];
        double[] orow = other.gradient[o];
        for (int j = 0; j < row.length; j++) {
          row[j] += orow[j];
        }
      }
    }

    AadResult toResult(AadTape tape, long paths, double seconds) {
      int no = value.length;
      double[] means = new double[no];
      double[] stderr = new double[no];
      double[][] grads = new double[no][];
      java.util.List<String> outNames = tape.outputNames();
      for (int o = 0; o < no; o++) {
        means[o] = value[o] / paths;
        if (paths > 1) {
          double var = (sumsq[o] - paths * means[o] * means[o]) / (paths - 1);
          stderr[o] = Math.sqrt(Math.max(0.0, var) / paths);
        } else {
          stderr[o] = Double.NaN;
        }
        double[] row = gradient[o].clone();
        for (int j = 0; j < row.length; j++) {
          row[j] /= paths;
        }
        grads[o] = row;
      }
      return AadResult.of(outNames, means, stderr, grads, tape.inputNames(), paths, seconds);
    }
  }
}
