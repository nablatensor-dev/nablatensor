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

import com.nablatensor.backend.cuda.CudaJit;

/**
 * Replays a tape as one kernel launch per node over an array of scenarios —
 * the shape an eager tensor framework produces when nothing fuses.
 *
 * <p>Also compilation-free, and included mainly because it is the arrangement
 * most people already have. It is the pessimistic bound for the same tape: each
 * node reads its operands from global memory and writes its result back, so
 * nothing is reused between adjacent operations and the whole tape's traffic is
 * paid at DRAM bandwidth. The interpreter differs only in doing the same walk
 * inside one kernel, and that difference is most of the performance.
 *
 * <p>The scenario count per pass is bounded by memory rather than by time: one
 * array per node means the buffers scale with tape size times scenarios, so a
 * 1,500-node tape in single precision costs about 6 KB per scenario before the
 * adjoint arrays double it.
 */
final class CudaEagerExecutable extends AbstractAadExecutable {

  private static final int BLOCK = CudaTapeKernels.BLOCK;
  private static final int GRID = 512;
  private static final long BUDGET_BYTES =
      Long.getLong("nablatensor.eager.bytes", 700L << 20);

  private final long forward;
  private final long reverse;
  private final long reduce;
  private final double compileSeconds;
  private final int nodes;
  private final int slot;
  private final long elementBytes;
  private final boolean adjoints;

  private final int[] ops;
  private final int[] argA;
  private final int[] argB;
  private final boolean[] active;
  private final int zeroIndex;
  private final int oneIndex;

  private final long constsBuffer;
  private final long inputsBuffer;
  private final long valuesBuffer;
  private final long adjointsBuffer;
  private final long partialsBuffer;

  CudaEagerExecutable(AadTape tape, AadOptions options) {
    super(tape, options);
    CudaTapeKernels.Handles handles = CudaTapeKernels.get(options.precision());
    this.forward = handles.eagerForward();
    this.reverse = handles.eagerReverse();
    this.reduce = handles.reduce();
    this.compileSeconds = handles.compileSeconds();
    this.nodes = tape.size();
    this.adjoints = options.adjoints();
    this.elementBytes = options.precision() == AadOptions.Precision.FLOAT32 ? 4 : 8;

    this.ops = new int[nodes];
    this.argA = new int[nodes];
    this.argB = new int[nodes];
    this.active = new boolean[nodes];
    // Two extra constants let a single launch of the forward kernel clear or
    // seed an adjoint row without needing a separate memset entry point.
    double[] constants = new double[nodes + 2];
    for (int i = 0; i < nodes; i++) {
      ops[i] = CudaTapeKernels.opcode(tape.op(i));
      argA[i] = tape.argA(i);
      argB[i] = tape.argB(i);
      active[i] = tape.isActive(i);
      constants[i] = tape.constant(i);
    }
    this.zeroIndex = nodes;
    this.oneIndex = nodes + 1;
    constants[zeroIndex] = 0.0;
    constants[oneIndex] = 1.0;

    long perScenario = (long) nodes * elementBytes * (adjoints ? 2 : 1);
    this.slot = (int) Math.max(BLOCK, Math.min(1 << 20, BUDGET_BYTES / perScenario));

    this.constsBuffer = CudaJit.malloc((long) constants.length * Double.BYTES);
    CudaJit.uploadDoubles(constsBuffer, constants);
    this.inputsBuffer = CudaJit.malloc((long) Math.max(1, inputs.length) * Double.BYTES);
    this.valuesBuffer = CudaJit.malloc((long) nodes * slot * elementBytes);
    this.adjointsBuffer = adjoints
        ? CudaJit.malloc((long) nodes * slot * elementBytes)
        : CudaJit.malloc(elementBytes);
    this.partialsBuffer = CudaJit.malloc((long) GRID * Double.BYTES);
  }

  @Override
  public String engineName() {
    return "cuda-eager";
  }

  @Override
  public double compileSeconds() {
    return compileSeconds;
  }

  @Override
  protected double defaultMaxChunkSeconds() {
    String override = System.getProperty("nablatensor.maxLaunchSeconds");
    if (override != null) {
      return Double.parseDouble(override);
    }
    return CudaJit.kernelTimeoutEnabled() ? 0.2 : 2.0;
  }

  private long row(long buffer, int node) {
    return buffer + (long) node * slot * elementBytes;
  }

  @Override
  public AadResult replay(long paths, long pathOffset, long seed) {
    checkOpen();
    if (paths <= 0) {
      throw new IllegalArgumentException("paths must be positive");
    }
    CudaJit.uploadDoubles(inputsBuffer, inputs);

    double value = 0.0;
    double[] gradients = new double[tape.inputCount()];
    long start = System.nanoTime();

    for (long done = 0; done < paths; done += slot) {
      long count = Math.min(slot, paths - done);
      long base = pathOffset + done;

      for (int i = 0; i < nodes; i++) {
        CudaJit.launch(forward, GRID, BLOCK,
            ops[i], i, argA[i], argA[i], base, seed,
            constsBuffer, inputsBuffer,
            argA[i] >= 0 ? row(valuesBuffer, argA[i]) : valuesBuffer,
            argB[i] >= 0 ? row(valuesBuffer, argB[i]) : valuesBuffer,
            row(valuesBuffer, i), count);
      }
      value += sum(row(valuesBuffer, tape.outputNode()), count);

      if (!adjoints) {
        continue;
      }
      // One launch clears every adjoint row, a second seeds the output's.
      CudaJit.launch(forward, GRID, BLOCK,
          CudaTapeKernels.opcode(AadOp.CONST), zeroIndex, 0, 0, base, seed,
          constsBuffer, inputsBuffer, adjointsBuffer, adjointsBuffer,
          adjointsBuffer, (long) nodes * slot);
      CudaJit.launch(forward, GRID, BLOCK,
          CudaTapeKernels.opcode(AadOp.CONST), oneIndex, 0, 0, base, seed,
          constsBuffer, inputsBuffer, adjointsBuffer, adjointsBuffer,
          row(adjointsBuffer, tape.outputNode()), count);

      for (int i = nodes - 1; i >= 0; i--) {
        if (!active[i]) {
          continue;
        }
        int a = argA[i];
        int b = argB[i];
        boolean liveA = a >= 0 && tape.isActive(a);
        boolean liveB = b >= 0 && tape.isActive(b);
        CudaJit.launch(reverse, GRID, BLOCK,
            ops[i],
            a >= 0 ? row(valuesBuffer, a) : valuesBuffer,
            b >= 0 ? row(valuesBuffer, b) : valuesBuffer,
            row(valuesBuffer, i),
            row(adjointsBuffer, i),
            liveA ? row(adjointsBuffer, a) : 0L,
            liveB ? row(adjointsBuffer, b) : 0L,
            count);
      }
      for (int j = 0; j < gradients.length; j++) {
        gradients[j] += sum(row(adjointsBuffer, tape.inputNode(j)), count);
      }
    }
    CudaJit.synchronize();
    double seconds = (System.nanoTime() - start) / 1e9;

    value /= paths;
    for (int j = 0; j < gradients.length; j++) {
      gradients[j] /= paths;
    }
    return new AadResult(value, gradients, tape.inputNames(), paths, seconds);
  }

  private double sum(long buffer, long count) {
    CudaJit.launch(reduce, GRID, BLOCK, buffer, count, partialsBuffer);
    CudaJit.synchronize();
    double[] partials = CudaJit.downloadDoubles(partialsBuffer, GRID);
    double total = 0.0;
    for (double partial : partials) {
      total += partial;
    }
    return total;
  }

  @Override
  public void close() {
    super.close();
    for (long buffer : new long[] {constsBuffer, inputsBuffer, valuesBuffer,
        adjointsBuffer, partialsBuffer}) {
      if (buffer != 0) {
        CudaJit.free(buffer);
      }
    }
  }
}
