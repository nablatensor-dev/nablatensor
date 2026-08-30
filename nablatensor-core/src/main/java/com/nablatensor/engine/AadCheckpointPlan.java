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

import java.util.ArrayList;
import java.util.List;

/**
 * Segment-checkpoint layout for a long tape, shared by the CUDA and Vulkan
 * adjoint code generators.
 *
 * <p>A fully unrolled reverse sweep keeps every forward value {@code v_i} live
 * until the reverse consumes it — for a 1500-node tape that is a live set no GPU
 * register file holds, so the shader spills to scratch memory and occupancy
 * collapses. This plan cuts the tape into {@code segments} contiguous pieces of
 * ~{@code sqrt(nodes)} nodes each. The forward runs piece by piece and, at each
 * internal boundary, saves only the handful of values that later nodes still
 * read (the <em>carry set</em>). The reverse then walks the pieces backwards:
 * for each it restores the boundary carry, recomputes that piece's forward into
 * a small local set, and runs the piece's reverse. Peak live set drops from
 * {@code O(nodes)} to {@code O(sqrt(nodes))} at the cost of recomputing the
 * forward once.
 *
 * <p>The carry set at boundary {@code b} is {@code C[b] = { i < b : some node
 * j >= b reads i as an argument }}. A carry node whose last use is at or beyond
 * the final internal boundary is <em>global</em>: it stays in a register for
 * the whole kernel body and is never written to the checkpoint buffer.
 */
public final class AadCheckpointPlan {

  public final int nodes;
  /** {@code bound[0] == 0}, {@code bound[segments] == nodes}, strictly increasing. */
  public final int[] bound;
  public final int segments;
  /** {@code lastUse[i]} = greatest node index that reads {@code i}; {@code nodes} for the output; {@code -1} if unread. */
  public final int[] lastUse;
  /** {@code carry[s]} = sorted {@code C[bound[s]]} for {@code s} in {@code 1..segments-1}; {@code carry[0]} empty. */
  public final int[][] carry;
  /** Nodes kept in a register for the whole kernel (carried across every internal boundary). */
  public final boolean[] global;
  /** Float offset into a path's checkpoint row where {@code carry[s] \ global} starts. */
  public final int[] sliceOffset;
  /** {@code carry[s] \ global}, i.e. the nodes segment {@code s} actually writes to / reads from the checkpoint buffer. */
  public final int[][] slice;
  /** Total floats per path in the checkpoint buffer. */
  public final int slotsPerPath;

  private AadCheckpointPlan(int nodes, int[] bound, int[] lastUse, int[][] carry,
                             boolean[] global, int[] sliceOffset, int[][] slice, int slotsPerPath) {
    this.nodes = nodes;
    this.bound = bound;
    this.segments = bound.length - 1;
    this.lastUse = lastUse;
    this.carry = carry;
    this.global = global;
    this.sliceOffset = sliceOffset;
    this.slice = slice;
    this.slotsPerPath = slotsPerPath;
  }

  public static boolean nodeArgA(AadOp op) {
    return switch (op) {
      case ADD, SUB, MUL, DIV, NEG, EXP, LOG, SQRT, ABS, MAX, MIN -> true;
      case CONST, INPUT, RANDN, RANDU -> false;
    };
  }

  public static boolean nodeArgB(AadOp op) {
    return switch (op) {
      case ADD, SUB, MUL, DIV, MAX, MIN -> true;
      default -> false;
    };
  }

  /**
   * Builds a plan, or returns {@code null} when checkpointing does not apply:
   * value-only kernels, tapes at or below {@code minNodes}, tapes that would
   * split into fewer than two segments, or tapes whose output node is not the
   * final node / is not differentiable (the unrolled path handles those, and
   * they are never the large tapes checkpointing exists for).
   */
  public static AadCheckpointPlan of(AadTape tape, AadOptions options, int minNodes) {
    if (!options.adjoints()) {
      return null;
    }
    int n = tape.size();
    if (n <= Math.max(2, minNodes)) {
      return null;
    }
    if (!tape.isActive(tape.outputNode())) {
      return null;
    }
    int segLen = Integer.getInteger("nablatensor.checkpoint.segLen",
        Math.max(48, (int) Math.round(Math.sqrt(n))));
    segLen = Math.max(4, segLen);
    int segCount = (n + segLen - 1) / segLen;
    if (segCount < 2) {
      return null;
    }
    int[] bound = new int[segCount + 1];
    for (int s = 1; s < segCount; s++) {
      bound[s] = Math.min(n, s * segLen);
    }
    bound[segCount] = n;
    // guard against a degenerate final boundary collision
    for (int s = 1; s <= segCount; s++) {
      if (bound[s] <= bound[s - 1]) {
        return null;
      }
    }

    int[] lastUse = new int[n];
    java.util.Arrays.fill(lastUse, -1);
    for (int j = 0; j < n; j++) {
      AadOp op = tape.op(j);
      if (nodeArgA(op)) {
        lastUse[tape.argA(j)] = j;
      }
      if (nodeArgB(op)) {
        lastUse[tape.argB(j)] = j;
      }
    }
    lastUse[tape.outputNode()] = n;

    if (tape.outputNode() < bound[segCount - 1]) {
      return null;
    }

    int lastInternal = bound[segCount - 1];
    boolean[] global = new boolean[n];
    for (int i = 0; i < n; i++) {
      global[i] = lastUse[i] >= lastInternal && i < lastInternal;
    }

    int[][] carry = new int[segCount][];
    carry[0] = new int[0];
    int[][] slice = new int[segCount][];
    int[] sliceOffset = new int[segCount];
    int running = 0;
    for (int s = 1; s < segCount; s++) {
      int b = bound[s];
      List<Integer> c = new ArrayList<>();
      List<Integer> sl = new ArrayList<>();
      for (int i = 0; i < b; i++) {
        if (lastUse[i] >= b) {
          c.add(i);
          if (!global[i]) {
            sl.add(i);
          }
        }
      }
      carry[s] = c.stream().mapToInt(Integer::intValue).toArray();
      slice[s] = sl.stream().mapToInt(Integer::intValue).toArray();
      sliceOffset[s] = running;
      running += slice[s].length;
    }

    return new AadCheckpointPlan(n, bound, lastUse, carry, global, sliceOffset, slice, running);
  }

  /** Union of every {@code carry[s]}: the nodes that need a persistent reverse accumulator. */
  public int[] carryUnion() {
    boolean[] seen = new boolean[nodes];
    int count = 0;
    for (int s = 1; s < segments; s++) {
      for (int i : carry[s]) {
        if (!seen[i]) {
          seen[i] = true;
          count++;
        }
      }
    }
    int[] out = new int[count];
    int k = 0;
    for (int i = 0; i < nodes; i++) {
      if (seen[i]) {
        out[k++] = i;
      }
    }
    return out;
  }

  /** The checkpoint slot for node {@code i} at boundary {@code s}, or {@code -1} if it is global / not in the slice. */
  public int slotOf(int s, int node) {
    int[] sl = slice[s];
    for (int k = 0; k < sl.length; k++) {
      if (sl[k] == node) {
        return sliceOffset[s] + k;
      }
    }
    return -1;
  }

  /** Value reference for node {@code i}: a whole-kernel register for a global carry node, a segment local otherwise. */
  public String vref(int i) {
    return (global[i] ? "g_v" : "v") + i;
  }

  /** Segment index containing node {@code i}. */
  public int segmentOf(int i) {
    int s = 0;
    while (bound[s + 1] <= i) {
      s++;
    }
    return s;
  }

  private boolean[] carryMembership;

  public boolean inCarry(int i) {
    if (carryMembership == null) {
      carryMembership = new boolean[nodes];
      for (int s = 1; s < segments; s++) {
        for (int node : carry[s]) {
          carryMembership[node] = true;
        }
      }
    }
    return carryMembership[i];
  }
}
