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

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;

/**
 * Single-precision sweep: twice the lanes of {@link VectorReplayF64} for the
 * same vector width, and half the working set for the same batch.
 *
 * <p>Both help, and neither is free. A long tape accumulates rounding across
 * the forward sweep and again across the adjoint sweep, so this is for
 * throughput on Greeks wanted to a few significant figures rather than for
 * reference valuation — the same trade the CUDA engine makes between its fp32
 * and fp64 kernels.
 *
 * <p>The lanes are single precision but the reductions are not. Summing
 * millions of scenarios in fp32 would lose more accuracy to the accumulation
 * than to the sweep, so scenario totals go into double accumulators, which
 * costs nothing measurable at one reduction per batch.
 *
 * <p>See {@link VectorReplayF64} for why every opcode has its own method.
 */
final class VectorReplayF32 extends BatchedReplay {

  private static final VectorSpecies<Float> SP = SimdSupport.FLOATS;
  private static final int LANES = SimdSupport.FLOAT_LANES;

  VectorReplayF32(AadTape tape, AadOptions options) {
    super(tape, options);
  }

  @Override
  public String engineName() {
    return "simd";
  }

  @Override
  Accumulator runRange(long pathFrom, long count, long seed, Draws crn) {
    final int n = ops.length;
    final float[] v = new float[n * BATCH];
    final float[] d = new float[n * BATCH];
    final double[] draws = new double[BATCH];
    final double[] in = inputs;
    final boolean adjoints = options.adjoints();
    final VectorPhilox rng = new VectorPhilox();
    final Accumulator acc = new Accumulator(inputRow.length);

    for (long done = 0; done < count; done += BATCH) {
      final int alive = (int) Math.min(BATCH, count - done);
      final long base = pathFrom + done;

      for (int i = 0; i < n; i++) {
        final int row = i * BATCH;
        final int a = rowA[i];
        final int b = rowB[i];
        switch (ops[i]) {
          case CONST -> Arrays.fill(v, row, row + BATCH, (float) constants[i]);
          case INPUT -> Arrays.fill(v, row, row + BATCH, (float) in[argA[i]]);
          case RANDN -> {
            // Drawn in double so the stream matches the other engines exactly,
            // then narrowed; the generator is a small share of the sweep.
            if (crn.read()) {
              System.arraycopy(crn.cache, crn.index(base, argA[i]), draws, 0, BATCH);
            } else {
              rng.normals(draws, base, seed, argA[i], BATCH);
              if (crn.write) {
                System.arraycopy(draws, 0, crn.cache, crn.index(base, argA[i]), BATCH);
              }
            }
            narrow(draws, v, row);
          }
          case ADD -> fwdAdd(v, a, b, row);
          case SUB -> fwdSub(v, a, b, row);
          case MUL -> fwdMul(v, a, b, row);
          case DIV -> fwdDiv(v, a, b, row);
          case NEG -> fwdNeg(v, a, row);
          case EXP -> fwdExp(v, a, row);
          case LOG -> fwdLog(v, a, row);
          case SQRT -> fwdSqrt(v, a, row);
          case ABS -> fwdAbs(v, a, row);
          case MAX -> fwdMax(v, a, b, row);
          case MIN -> fwdMin(v, a, b, row);
        }
      }
      for (int p = 0; p < alive; p++) {
        acc.value += v[outRow + p];
      }

      if (!adjoints) {
        continue;
      }
      Arrays.fill(d, 0.0f);
      Arrays.fill(d, outRow, outRow + BATCH, 1.0f);

      for (int i = n - 1; i >= 0; i--) {
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

  private static void narrow(double[] source, float[] target, int row) {
    for (int p = 0; p < BATCH; p++) {
      target[row + p] = (float) source[p];
    }
  }

  private static FloatVector ld(float[] array, int index) {
    return FloatVector.fromArray(SP, array, index);
  }

  private static void fwdAdd(float[] v, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).add(ld(v, b + p)).intoArray(v, row + p);
    }
  }

  private static void fwdSub(float[] v, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).sub(ld(v, b + p)).intoArray(v, row + p);
    }
  }

  private static void fwdMul(float[] v, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).mul(ld(v, b + p)).intoArray(v, row + p);
    }
  }

  private static void fwdDiv(float[] v, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).div(ld(v, b + p)).intoArray(v, row + p);
    }
  }

  private static void fwdNeg(float[] v, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).neg().intoArray(v, row + p);
    }
  }

  private static void fwdExp(float[] v, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).lanewise(VectorOperators.EXP).intoArray(v, row + p);
    }
  }

  private static void fwdLog(float[] v, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).lanewise(VectorOperators.LOG).intoArray(v, row + p);
    }
  }

  private static void fwdSqrt(float[] v, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).lanewise(VectorOperators.SQRT).intoArray(v, row + p);
    }
  }

  private static void fwdAbs(float[] v, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).abs().intoArray(v, row + p);
    }
  }

  private static void fwdMax(float[] v, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).max(ld(v, b + p)).intoArray(v, row + p);
    }
  }

  private static void fwdMin(float[] v, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(v, a + p).min(ld(v, b + p)).intoArray(v, row + p);
    }
  }

  private static void revAdd(float[] d, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      FloatVector adj = ld(d, row + p);
      ld(d, a + p).add(adj).intoArray(d, a + p);
      ld(d, b + p).add(adj).intoArray(d, b + p);
    }
  }

  private static void revSub(float[] d, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      FloatVector adj = ld(d, row + p);
      ld(d, a + p).add(adj).intoArray(d, a + p);
      ld(d, b + p).sub(adj).intoArray(d, b + p);
    }
  }

  private static void revMul(float[] v, float[] d, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      FloatVector adj = ld(d, row + p);
      ld(d, a + p).add(adj.mul(ld(v, b + p))).intoArray(d, a + p);
      ld(d, b + p).add(adj.mul(ld(v, a + p))).intoArray(d, b + p);
    }
  }

  private static void revDiv(float[] v, float[] d, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      FloatVector adj = ld(d, row + p);
      FloatVector den = ld(v, b + p);
      ld(d, a + p).add(adj.div(den)).intoArray(d, a + p);
      ld(d, b + p).sub(adj.mul(ld(v, row + p)).div(den)).intoArray(d, b + p);
    }
  }

  private static void revNeg(float[] d, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(d, a + p).sub(ld(d, row + p)).intoArray(d, a + p);
    }
  }

  private static void revExp(float[] v, float[] d, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(d, a + p).add(ld(d, row + p).mul(ld(v, row + p))).intoArray(d, a + p);
    }
  }

  private static void revLog(float[] v, float[] d, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(d, a + p).add(ld(d, row + p).div(ld(v, a + p))).intoArray(d, a + p);
    }
  }

  private static void revSqrt(float[] v, float[] d, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      ld(d, a + p).add(ld(d, row + p).mul(0.5f).div(ld(v, row + p))).intoArray(d, a + p);
    }
  }

  private static void revAbs(float[] v, float[] d, int a, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      FloatVector adj = ld(d, row + p);
      VectorMask<Float> negative = ld(v, a + p).compare(VectorOperators.LT, 0.0f);
      ld(d, a + p).add(adj.blend(adj.neg(), negative)).intoArray(d, a + p);
    }
  }

  private static void revMax(float[] v, float[] d, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      FloatVector adj = ld(d, row + p);
      VectorMask<Float> takeA = ld(v, a + p).compare(VectorOperators.GE, ld(v, b + p));
      ld(d, a + p).add(adj, takeA).intoArray(d, a + p);
      ld(d, b + p).add(adj, takeA.not()).intoArray(d, b + p);
    }
  }

  private static void revMin(float[] v, float[] d, int a, int b, int row) {
    for (int p = 0; p < BATCH; p += LANES) {
      FloatVector adj = ld(d, row + p);
      VectorMask<Float> takeA = ld(v, a + p).compare(VectorOperators.LE, ld(v, b + p));
      ld(d, a + p).add(adj, takeA).intoArray(d, a + p);
      ld(d, b + p).add(adj, takeA.not()).intoArray(d, b + p);
    }
  }
}
