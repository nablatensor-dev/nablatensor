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

/**
 * Single-threaded reference replay, kept as a static entry point because its
 * job is verification rather than throughput: it evaluates the same tape over
 * the same path stream as an accelerated engine, sharing none of that engine's
 * code, so agreement between the two is evidence rather than tautology.
 */
public final class AadCpuReplay {

  private AadCpuReplay() {
  }

  public static AadResult replay(AadTape tape, double[] inputs, long paths, long seed) {
    AadOptions options = new AadOptions(AadOptions.Precision.FLOAT64, true, 1);
    try (ScalarReplay replay = new ScalarReplay(tape, options)) {
      for (int j = 0; j < inputs.length; j++) {
        replay.setInput(tape.inputNames().get(j), inputs[j]);
      }
      return replay.replay(paths, 0L, seed);
    }
  }
}
