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
import java.util.Arrays;
import java.util.List;

/**
 * A recorded scalar computation graph: the "kernel" in kernel-AAD terms.
 *
 * <p>Nodes are appended in evaluation order, so the array order is already a
 * valid topological order for the forward sweep and its reverse is a valid
 * order for the adjoint sweep. Recording happens once; the tape is then
 * translated into a device kernel and replayed for every scenario.
 */
public final class AadTape {

  private final AadOp[] op;
  private final int[] argA;
  private final int[] argB;
  private final double[] constant;
  private final int[] inputNode;
  private final String[] inputName;
  private final double[] recordedInput;
  private final String[] randStreamName;
  private final int[] randNormalCount;   // per stream
  private final int[] randUniformCount;  // per stream
  private final int[] outputNode;
  private final String[] outputName;
  private final boolean[] active;

  AadTape(AadOp[] op, int[] argA, int[] argB, double[] constant,
           int[] inputNode, String[] inputName, double[] recordedInput,
           String[] randStreamName, int[] randNormalCount, int[] randUniformCount,
           int[] outputNode, String[] outputName) {
    this.op = op;
    this.argA = argA;
    this.argB = argB;
    this.constant = constant;
    this.inputNode = inputNode;
    this.inputName = inputName;
    this.recordedInput = recordedInput;
    this.randStreamName = randStreamName;
    this.randNormalCount = randNormalCount;
    this.randUniformCount = randUniformCount;
    this.outputNode = outputNode;
    this.outputName = outputName;
    this.active = markActive();
  }

  /**
   * A node is active when its value depends on at least one differentiable
   * input. Only active nodes need an adjoint slot, which typically removes the
   * whole random-number and constant sub-graph from the reverse sweep.
   */
  private boolean[] markActive() {
    boolean[] flags = new boolean[op.length];
    for (int i = 0; i < op.length; i++) {
      flags[i] = switch (op[i]) {
        case INPUT -> true;
        case CONST, RANDN, RANDU -> false;
        case NEG, EXP, LOG, SQRT, ABS -> flags[argA[i]];
        default -> flags[argA[i]] || flags[argB[i]];
      };
    }
    return flags;
  }

  public int size() {
    return op.length;
  }

  public AadOp op(int node) {
    return op[node];
  }

  public int argA(int node) {
    return argA[node];
  }

  public int argB(int node) {
    return argB[node];
  }

  public double constant(int node) {
    return constant[node];
  }

  public boolean isActive(int node) {
    return active[node];
  }

  public int inputCount() {
    return inputNode.length;
  }

  public int inputNode(int index) {
    return inputNode[index];
  }

  public String inputName(int index) {
    return inputName[index];
  }

  public List<String> inputNames() {
    return List.of(inputName);
  }

  /** Input values as seen during recording; the starting point for a replay. */
  public double[] recordedInputs() {
    return recordedInput.clone();
  }

  /** Normal draws per scenario on the default stream — the common single-stream total. */
  public int randCount() {
    return randNormalCount[0];
  }

  /** Number of independent random streams; {@code 1} unless {@code rec.stream(name)} was used. */
  public int randStreamCount() {
    return randStreamName.length;
  }

  public String randStreamName(int stream) {
    return randStreamName[stream];
  }

  /** Standard-normal draws per scenario on {@code stream}. */
  public int randNormalCount(int stream) {
    return randNormalCount[stream];
  }

  /** Uniform {@code [0,1)} draws per scenario on {@code stream}. */
  public int randUniformCount(int stream) {
    return randUniformCount[stream];
  }

  /** For a {@code RANDN}/{@code RANDU} node: which stream it draws from. */
  public int randStreamOf(int node) {
    return argB[node];
  }

  /** For a {@code RANDN}/{@code RANDU} node: its 0-based index within its (stream, kind). */
  public int randOrdinal(int node) {
    return argA[node];
  }

  /** Total random draws per scenario across every stream and both kinds. */
  public int randTotal() {
    int total = 0;
    for (int s = 0; s < randStreamName.length; s++) {
      total += randNormalCount[s] + randUniformCount[s];
    }
    return total;
  }

  /**
   * Flat-buffer offset of {@code stream}'s draws in the layout
   * {@code [s0 normals | s0 uniforms | s1 normals | ...]}.
   */
  public int randStreamOffset(int stream) {
    int off = 0;
    for (int s = 0; s < stream; s++) {
      off += randNormalCount[s] + randUniformCount[s];
    }
    return off;
  }

  /** Flat-buffer index of a {@code RANDN}/{@code RANDU} node's draw. */
  public int randFlatIndex(int node) {
    int stream = argB[node];
    int base = randStreamOffset(stream);
    return op[node] == AadOp.RANDU ? base + randNormalCount[stream] + argA[node] : base + argA[node];
  }

  /** Whether the tape uses anything beyond a single stream of standard-normal draws. */
  public boolean hasExtendedRandom() {
    if (randStreamName.length > 1) {
      return true;
    }
    return randUniformCount[0] > 0;
  }

  /** Node index of the primary (first-recorded) output. */
  public int outputNode() {
    return outputNode[0];
  }

  /** Number of recorded outputs; {@code 1} for a single {@code rec.output(...)}. */
  public int outputCount() {
    return outputNode.length;
  }

  /** Node index of output {@code i}, in recording order. */
  public int outputNode(int i) {
    return outputNode[i];
  }

  /** Name of output {@code i}; {@code "value"} for a single unnamed output. */
  public String outputName(int i) {
    return outputName[i];
  }

  public List<String> outputNames() {
    return List.of(outputName);
  }

  @Override
  public String toString() {
    return "AadTape(nodes=" + op.length + ", inputs=" + Arrays.toString(inputName)
        + ", randoms=" + randCount() + ")";
  }

  /** Growable builder used by {@link AadRecorder}. */
  static final class Builder {
    private AadOp[] op = new AadOp[256];
    private int[] argA = new int[256];
    private int[] argB = new int[256];
    private double[] constant = new double[256];
    private int size;
    private final List<Integer> inputNodes = new ArrayList<>();
    private final List<String> inputNames = new ArrayList<>();
    private final List<Double> inputValues = new ArrayList<>();
    private final List<String> randStreams = new ArrayList<>(List.of("default"));
    private final List<int[]> randCounts = new ArrayList<>(List.of(new int[2])); // [normal, uniform] per stream
    private final List<Integer> outputNodes = new ArrayList<>();
    private final List<String> outputNames = new ArrayList<>();

    int add(AadOp operation, int a, int b, double value) {
      if (size == op.length) {
        int grown = size * 2;
        op = Arrays.copyOf(op, grown);
        argA = Arrays.copyOf(argA, grown);
        argB = Arrays.copyOf(argB, grown);
        constant = Arrays.copyOf(constant, grown);
      }
      op[size] = operation;
      argA[size] = a;
      argB[size] = b;
      constant[size] = value;
      return size++;
    }

    int addInput(String name, double value) {
      int node = add(AadOp.INPUT, inputNodes.size(), -1, value);
      inputNodes.add(node);
      inputNames.add(name);
      inputValues.add(value);
      return node;
    }

    private int streamIndex(String name) {
      for (int i = 0; i < randStreams.size(); i++) {
        if (randStreams.get(i).equals(name)) {
          return i;
        }
      }
      randStreams.add(name);
      randCounts.add(new int[2]);
      return randStreams.size() - 1;
    }

    int addRandn() {
      return addRandn("default");
    }

    int addRandn(String stream) {
      int si = streamIndex(stream);
      return add(AadOp.RANDN, randCounts.get(si)[0]++, si, 0.0);
    }

    int addRandu(String stream) {
      int si = streamIndex(stream);
      return add(AadOp.RANDU, randCounts.get(si)[1]++, si, 0.0);
    }

    void setOutput(int node) {
      addOutput("value", node);
    }

    void addOutput(String name, int node) {
      if (outputNames.contains(name)) {
        throw new IllegalArgumentException("output '" + name + "' recorded twice");
      }
      outputNames.add(name);
      outputNodes.add(node);
    }

    AadTape build() {
      if (outputNodes.isEmpty()) {
        throw new IllegalStateException("no output was recorded");
      }
      int[] nodes = new int[inputNodes.size()];
      double[] values = new double[inputNodes.size()];
      for (int i = 0; i < nodes.length; i++) {
        nodes[i] = inputNodes.get(i);
        values[i] = inputValues.get(i);
      }
      int[] outNodes = new int[outputNodes.size()];
      for (int i = 0; i < outNodes.length; i++) {
        outNodes[i] = outputNodes.get(i);
      }
      int[] normal = new int[randStreams.size()];
      int[] uniform = new int[randStreams.size()];
      for (int i = 0; i < randStreams.size(); i++) {
        normal[i] = randCounts.get(i)[0];
        uniform[i] = randCounts.get(i)[1];
      }
      return new AadTape(Arrays.copyOf(op, size), Arrays.copyOf(argA, size), Arrays.copyOf(argB, size),
          Arrays.copyOf(constant, size), nodes, inputNames.toArray(String[]::new), values,
          randStreams.toArray(String[]::new), normal, uniform,
          outNodes, outputNames.toArray(String[]::new));
    }
  }
}
