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
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorSpecies;

/** Vector API facts, isolated so nothing resolves these types before the module is known to exist. */
final class SimdSupport {

  static final VectorSpecies<Double> DOUBLES = DoubleVector.SPECIES_PREFERRED;
  static final VectorSpecies<Float> FLOATS = FloatVector.SPECIES_PREFERRED;
  static final VectorSpecies<Long> LONGS = LongVector.SPECIES_PREFERRED;
  static final int DOUBLE_LANES = DOUBLES.length();
  static final int FLOAT_LANES = FLOATS.length();

  /**
   * Scenarios held in flight per sweep.
   *
   * <p>This is the knob that decides whether the vector units are reachable at
   * all. The sweeps dispatch on each node's opcode, and that dispatch — an
   * indirect branch the hardware cannot predict, since the opcode sequence is
   * whatever the tape happened to record — costs far more than the arithmetic
   * it guards. Evaluating a node for a single vector of scenarios pays the
   * dispatch once per vector; evaluating it for {@code BATCH} scenarios pays it
   * once per {@code BATCH / lanes} vectors instead, which is the reason this
   * engine is faster than the scalar one rather than merely wider.
   *
   * <p>The cost is working set: two arrays of {@code nodes * BATCH} per worker.
   * That has to be read alongside the thread count, because every worker keeps
   * its own pair. At 64 a 1,500-node fp64 tape needs about 1.5 MB per worker,
   * which twelve workers turn into 18 MB against a 12 MB L3 — measurably worse
   * than 32, where the same twelve fit with room to spare. 32 gives up a little
   * single-threaded throughput to keep the parallel case in cache; raise it for
   * small tapes or few threads.
   */
  static final int BATCH = Integer.getInteger("nablatensor.simd.batch", 32);

  private SimdSupport() {
  }

  static {
    if (BATCH % FLOAT_LANES != 0 || BATCH % DOUBLE_LANES != 0) {
      throw new ExceptionInInitializerError(
          "nablatensor.simd.batch must be a multiple of " + FLOAT_LANES + " and " + DOUBLE_LANES);
    }
  }

  static String describe() {
    return DOUBLES.vectorShape() + ", " + DOUBLE_LANES + "x fp64 / " + FLOAT_LANES
        + "x fp32 per vector, batch " + BATCH;
  }
}
