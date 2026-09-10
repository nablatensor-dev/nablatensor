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
 * A recorded tape copied into flat parallel arrays, once, at compile time.
 *
 * <p>Every host engine walks the same tape millions of times. Calling
 * {@link AadTape#op(int)} and friends per node per scenario costs a bounds
 * check and a field load that no amount of JIT will hoist out of the scenario
 * loop, so each engine used to flatten the tape into its own set of arrays in
 * its own constructor. This is that copy, written once.
 *
 * <p>Fields are final arrays rather than accessors deliberately: the sweeps
 * read them in their innermost loop, and a plain array read is what C2 turns
 * into a single load.
 *
 * <p>Public only so the engine modules can use it; not part of the supported
 * API.
 */
@Internal
public final class FlatTape {

  /** Opcode per node. */
  public final AadOp[] op;
  /** First argument per node: a node index, or an input index for {@code INPUT}. */
  public final int[] argA;
  /** Second argument per node: a node index, or a stream index for the draws. */
  public final int[] argB;
  /** Literal value for {@code CONST} nodes, zero elsewhere. */
  public final double[] constant;
  /** Whether a node is on a path from some input to some output. */
  public final boolean[] active;
  /** Node index of each differentiable input, in input order. */
  public final int[] inputNode;
  /** Node index of each recorded output, in recording order. */
  public final int[] outputNode;
  /** Number of independent random streams the tape draws from. */
  public final int randStreams;

  public FlatTape(AadTape tape) {
    int n = tape.size();
    this.op = new AadOp[n];
    this.argA = new int[n];
    this.argB = new int[n];
    this.constant = new double[n];
    this.active = new boolean[n];
    for (int i = 0; i < n; i++) {
      op[i] = tape.op(i);
      argA[i] = tape.argA(i);
      argB[i] = tape.argB(i);
      constant[i] = tape.constant(i);
      active[i] = tape.isActive(i);
    }
    this.inputNode = new int[tape.inputCount()];
    for (int j = 0; j < inputNode.length; j++) {
      inputNode[j] = tape.inputNode(j);
    }
    this.outputNode = new int[tape.outputCount()];
    for (int o = 0; o < outputNode.length; o++) {
      outputNode[o] = tape.outputNode(o);
    }
    this.randStreams = tape.randStreamCount();
  }

  public int size() {
    return op.length;
  }

  /**
   * {@code nodes} multiplied by {@code stride}, for a sweep that stores
   * {@code stride} scenarios side by side per node and therefore indexes rows
   * rather than nodes.
   */
  public static int[] scaled(int[] nodes, int stride) {
    int[] rows = new int[nodes.length];
    for (int i = 0; i < nodes.length; i++) {
      rows[i] = nodes[i] * stride;
    }
    return rows;
  }
}
