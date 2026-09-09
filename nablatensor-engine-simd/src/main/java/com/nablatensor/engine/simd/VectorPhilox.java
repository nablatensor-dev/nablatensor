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

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Philox2x32-10 for a whole batch of scenarios, split between scalar and vector
 * work along the line where each actually pays.
 *
 * <p>Counter-based generation is what makes the batched sweeps possible. Every
 * draw is a pure function of {@code (path, drawIndex)}, so a node needing its
 * draw for a batch of scenarios can compute them directly, with no per-lane
 * state to carry and nothing to serialise. A sequential generator would force
 * the sweep back to one scenario at a time.
 *
 * <p>The ten mixing rounds run in scalar integer arithmetic. Expressed with
 * vector lanes they unroll into several hundred live vector temporaries, past
 * the point where escape analysis will track them, and every one becomes a heap
 * allocation — measured at 7.2 KB per call, which was 78% of everything this
 * engine allocated. The rounds are cheap integer operations the hardware issues
 * several of per cycle anyway, so little is lost. Box-Muller is the opposite
 * case: a logarithm, a square root and a sine or cosine per draw, where the
 * vector library is worth several times the scalar routines, so that part stays
 * vectorised over a short loop escape analysis handles easily.
 *
 * <p>The pairing convention reproduces the CUDA kernel's stream exactly. There
 * a Box-Muller pair is produced per counter and the sine half cached, so draw
 * {@code k} is counter {@code k/2}, taking the cosine when {@code k} is even
 * and the sine when it is odd.
 */
final class VectorPhilox {

  private static final VectorSpecies<Double> SP = SimdSupport.DOUBLES;
  private static final int LANES = SimdSupport.DOUBLE_LANES;
  private static final double UINT_SCALE = 2.3283064365386963e-10;

  // Allocated once per worker so a draw costs no allocation at all.
  private final double[] u1 = new double[SimdSupport.BATCH];
  private final double[] u2 = new double[SimdSupport.BATCH];
  // The sine half of the last Box-Muller pair, kept so that the odd draw of a
  // counter is a copy rather than a second Philox block + log + sqrt.
  private final double[] sinHalf = new double[SimdSupport.BATCH];
  private long cachedBase = Long.MIN_VALUE;
  private int cachedCounter = -1;

  /**
   * Fills {@code out[0 .. count)} with draw {@code drawIndex} for scenarios
   * {@code basePath .. basePath+count}.
   *
   * <p>Each Philox counter yields a Box-Muller pair — cosine half for the even
   * draw, sine half for the odd one. The even draw computes both halves and
   * stashes the sine one; the odd draw of the same counter then just copies it,
   * so the Philox mixing, the vectorised {@code log}/{@code sqrt} and one
   * {@code sin}/{@code cos} run once per two draws instead of once per draw.
   */
  void normals(double[] out, long basePath, long seed, int drawIndex, int count) {
    final int counter = drawIndex >>> 1;
    final boolean odd = (drawIndex & 1) != 0;

    if (odd && counter == cachedCounter && basePath == cachedBase) {
      System.arraycopy(sinHalf, 0, out, 0, count);
      return;
    }

    final int seedLo = (int) seed;
    final int seedHi = (int) (seed >>> 32);
    final double[] lowWords = u1;
    final double[] highWords = u2;

    for (int p = 0; p < count; p++) {
      final long path = basePath + p;
      int c0 = (int) path ^ seedLo;
      int c1 = (int) (path >>> 32) ^ seedHi ^ counter;
      int key = 0x1BD11BDA;
      for (int round = 0; round < 10; round++) {
        long product = 0xD256D193L * Integer.toUnsignedLong(c0);
        int high = (int) (product >>> 32);
        int low = (int) product;
        c0 = high ^ key ^ c1;
        c1 = low;
        key += 0x9E3779B9;
      }
      lowWords[p] = (Integer.toUnsignedLong(c0) + 0.5) * UINT_SCALE;
      highWords[p] = (Integer.toUnsignedLong(c1) + 0.5) * UINT_SCALE;
    }

    for (int p = 0; p < count; p += LANES) {
      DoubleVector radius = DoubleVector.fromArray(SP, lowWords, p)
          .lanewise(VectorOperators.LOG)
          .mul(-2.0)
          .lanewise(VectorOperators.SQRT);
      DoubleVector angle = DoubleVector.fromArray(SP, highWords, p).mul(6.283185307179586);
      radius.mul(angle.lanewise(VectorOperators.COS)).intoArray(out, p);       // cosine half
      radius.mul(angle.lanewise(VectorOperators.SIN)).intoArray(sinHalf, p);   // sine half, cached
    }
    cachedBase = basePath;
    cachedCounter = counter;
    if (odd) {
      System.arraycopy(sinHalf, 0, out, 0, count);
    }
  }
}
