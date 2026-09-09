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
package com.nablatensor.engine.simd;

import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadTape;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;

/**
 * Double-precision sweep, one scenario per lane and {@link #BATCH} scenarios in
 * flight.
 *
 * <p>Every opcode's loop lives in its own small method, and that is load
 * bearing rather than stylistic. C2 keeps a {@code DoubleVector} in registers
 * only if escape analysis proves it never escapes, and escape analysis gives up
 * on methods past a certain size. With all fourteen opcodes and both sweeps in
 * one body it bailed and every node allocated — measured at 88 bytes per node,
 * 136 KB per scenario, which is why the first version of this engine lost to
 * the scalar one. Split into per-opcode methods, the same arithmetic allocates
 * nothing.
 *
 * <p>The same limit is why {@link #forward} and {@link #reverse} are separate
 * methods rather than two loops in {@link #runRange}. Held together they made
 * {@code runRange} 1,023 bytes of bytecode, and folding fourteen opcode helpers
 * and the generator into a body that size exhausts what C2 will inline in one
 * pass — eight lanes to a vector means each helper unrolls to four vector
 * operations, twice what the single-precision sweep expands to — so the rest is
 * deferred to incremental inlining. Vector calls resolved that late lose the
 * constant folding the Vector API relies on: the operator argument of
 * {@code lanewise} is no longer a constant when the branch that routes
 * transcendentals to the vector math library is parsed, so {@code EXP} takes
 * the general path instead and reaches code that does not expect a math-library
 * opcode. On JDK 25 that aborts the compiler thread outright, and roughly a
 * quarter of fresh JVMs running the adjoint sweep died that way; short of the
 * crash it costs throughput and makes it depend on which compilation happened
 * to win, which is how the fault first showed up — a benchmark row quotable
 * only as a range. Split, each sweep is compiled on its own instead of through
 * a driver too large to carry it, which is both stable and faster than the one
 * body ever was.
 */
final class VectorReplayF64 extends BatchedReplay {

  private static final VectorSpecies<Double> SP = SimdSupport.DOUBLES;
  private static final int LANES = SimdSupport.DOUBLE_LANES;
  // Benchmarking probes (not numerically correct): isolate the RNG / exp share.
  private static final boolean SKIP_RNG = "zero".equals(System.getProperty("nablatensor.simd.randn"));
  private static final boolean SKIP_EXP = "none".equals(System.getProperty("nablatensor.simd.exp"));

  private final boolean adjoints;

  VectorReplayF64(AadTape tape, AadOptions options) {
    super(tape, options);
    this.adjoints = options.adjoints();
  }

  @Override
  public String engineName() {
    return "simd";
  }

  @Override
  Accumulator runRange(long pathFrom, long count, long seed, Draws crn) {
    final int n = ops.length;
    final double[] v = new double[n * BATCH];
    final double[] d = adjoints ? new double[n * BATCH] : null;
    final double[] draws = new double[BATCH];
    final VectorPhilox rng = new VectorPhilox();
    final Accumulator acc = new Accumulator(inputRow.length);

    for (long done = 0; done < count; done += BATCH) {
      final int alive = (int) Math.min(BATCH, count - done);
      final long base = pathFrom + done;

      forward(v, draws, rng, base, seed, crn);
      for (int p = 0; p < alive; p++) {
        acc.value += v[outRow + p];
      }

      if (!adjoints) {
        continue;
      }
      Arrays.fill(d, 0.0);
      Arrays.fill(d, outRow, outRow + BATCH, 1.0);
      reverse(v, d);

      for (int j = 0; j < acc.gradient.length; j++) {
        final int from = inputRow[j];
        double sum = 0.0;
        for (int p = 0; p < alive; p++) {
          sum += d[from + p];
        }
        acc.gradient[j] += sum;
      }
    }
    return acc;
  }

  private void forward(double[] v, double[] draws, VectorPhilox rng, long base, long seed, Draws crn) {
    final double[] in = inputs;
    for (int i = 0, n = ops.length; i < n; i++) {
      final int row = i * BATCH;
      final int a = rowA[i];
      final int b = rowB[i];
      switch (ops[i]) {
        case CONST -> Arrays.fill(v, row, row + BATCH, constants[i]);
        case INPUT -> Arrays.fill(v, row, row + BATCH, in[argA[i]]);
        case RANDN -> {
          if (crn.read()) {
            System.arraycopy(crn.cache, crn.index(base, argA[i]), v, row, BATCH);
          } else if (!SKIP_RNG) {
            rng.normals(draws, base, seed, argA[i], BATCH);
            System.arraycopy(draws, 0, v, row, BATCH);
            if (crn.write) {
              System.arraycopy(draws, 0, crn.cache, crn.index(base, argA[i]), BATCH);
            }
          }
        }
        case ADD -> fwdAdd(v, a, b, row);
        case SUB -> fwdSub(v, a, b, row);
        case MUL -> fwdMul(v, a, b, row);
        case DIV -> fwdDiv(v, a, b, row);
        case NEG -> fwdNeg(v, a, row);
        case EXP -> { if (!SKIP_EXP) fwdExp(v, a, row); else System.arraycopy(v, a, v, row, BATCH); }
        case LOG -> fwdLog(v, a, row);
        case SQRT -> fwdSqrt(v, a, row);
        case ABS -> fwdAbs(v, a, row);
        case MAX -> fwdMax(v, a, b, row);
        case MIN -> fwdMin(v, a, b, row);
      }
    }
  }

  private void reverse(double[] v, double[] d) {
    for (int i = ops.length - 1; i >= 0; i--) {
      if (!active[i]) {
        continue;
      }
      final int row = i * BATCH;
      final int a = rowA[i];
      final int b = rowB[i];
      switch (ops[i]) {
        case CONST, INPUT, RANDN, RANDU -> {
        }
        case ADD -> revAdd(d, a, b, row);
        case SUB -> revSub(d, a, b, row);
        case MUL -> revMul(v, d, a, b, row);
        case DIV -> revDiv(v, d, a, b, row);
        case NEG -> revNeg(d, a, row);
        case EXP -> revExp(v, d, a, row);
        case LOG -> revLog(v, d, a, row);
        case SQRT -> revSqrt(v, d, a, row);
        case ABS -> revAbs(v, d, a, row);
        case MAX -> revMax(v, d, a, b, row);
        case MIN -> revMin(v, d, a, b, row);
      }
    }
  }

  private static DoubleVector ld(double[] array, int index) {
    return DoubleVector.fromArray(SP, array, index);
  }

  private static void fwdAdd(double[] v, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).add(ld(v, b + p)).intoArray(v, row + p);
    }
  }

  private static void fwdSub(double[] v, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).sub(ld(v, b + p)).intoArray(v, row + p);
    }
  }

  private static void fwdMul(double[] v, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).mul(ld(v, b + p)).intoArray(v, row + p);
    }
  }

  private static void fwdDiv(double[] v, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).div(ld(v, b + p)).intoArray(v, row + p);
    }
  }

  private static void fwdNeg(double[] v, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).neg().intoArray(v, row + p);
    }
  }

  private static void fwdExp(double[] v, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).lanewise(VectorOperators.EXP).intoArray(v, row + p);
    }
  }

  private static void fwdLog(double[] v, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).lanewise(VectorOperators.LOG).intoArray(v, row + p);
    }
  }

  private static void fwdSqrt(double[] v, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).lanewise(VectorOperators.SQRT).intoArray(v, row + p);
    }
  }

  private static void fwdAbs(double[] v, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).abs().intoArray(v, row + p);
    }
  }

  private static void fwdMax(double[] v, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).max(ld(v, b + p)).intoArray(v, row + p);
    }
  }

  private static void fwdMin(double[] v, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).min(ld(v, b + p)).intoArray(v, row + p);
    }
  }

  private static void revAdd(double[] d, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      DoubleVector adj = ld(d, row + p);
      ld(d, a + p).add(adj).intoArray(d, a + p);
      ld(d, b + p).add(adj).intoArray(d, b + p);
    }
  }

  private static void revSub(double[] d, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      DoubleVector adj = ld(d, row + p);
      ld(d, a + p).add(adj).intoArray(d, a + p);
      ld(d, b + p).sub(adj).intoArray(d, b + p);
    }
  }

  private static void revMul(double[] v, double[] d, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      DoubleVector adj = ld(d, row + p);
      ld(d, a + p).add(adj.mul(ld(v, b + p))).intoArray(d, a + p);
      ld(d, b + p).add(adj.mul(ld(v, a + p))).intoArray(d, b + p);
    }
  }

  private static void revDiv(double[] v, double[] d, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      DoubleVector adj = ld(d, row + p);
      DoubleVector den = ld(v, b + p);
      ld(d, a + p).add(adj.div(den)).intoArray(d, a + p);
      ld(d, b + p).sub(adj.mul(ld(v, row + p)).div(den)).intoArray(d, b + p);
    }
  }

  private static void revNeg(double[] d, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(d, a + p).sub(ld(d, row + p)).intoArray(d, a + p);
    }
  }

  private static void revExp(double[] v, double[] d, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(d, a + p).add(ld(d, row + p).mul(ld(v, row + p))).intoArray(d, a + p);
    }
  }

  private static void revLog(double[] v, double[] d, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(d, a + p).add(ld(d, row + p).div(ld(v, a + p))).intoArray(d, a + p);
    }
  }

  private static void revSqrt(double[] v, double[] d, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(d, a + p).add(ld(d, row + p).mul(0.5).div(ld(v, row + p))).intoArray(d, a + p);
    }
  }

  private static void revAbs(double[] v, double[] d, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      DoubleVector adj = ld(d, row + p);
      VectorMask<Double> negative = ld(v, a + p).compare(VectorOperators.LT, 0.0);
      ld(d, a + p).add(adj.blend(adj.neg(), negative)).intoArray(d, a + p);
    }
  }

  // The branch is chosen per lane, so each scenario's adjoint follows the side
  // that scenario actually took.
  private static void revMax(double[] v, double[] d, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      DoubleVector adj = ld(d, row + p);
      VectorMask<Double> takeA = ld(v, a + p).compare(VectorOperators.GE, ld(v, b + p));
      ld(d, a + p).add(adj, takeA).intoArray(d, a + p);
      ld(d, b + p).add(adj, takeA.not()).intoArray(d, b + p);
    }
  }

  private static void revMin(double[] v, double[] d, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      DoubleVector adj = ld(d, row + p);
      VectorMask<Double> takeA = ld(v, a + p).compare(VectorOperators.LE, ld(v, b + p));
      ld(d, a + p).add(adj, takeA).intoArray(d, a + p);
      ld(d, b + p).add(adj, takeA.not()).intoArray(d, b + p);
    }
  }
}
