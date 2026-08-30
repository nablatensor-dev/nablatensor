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
 * Replays a tape on the GPU by interpreting it, with no per-tape compilation.
 *
 * <p>The tape is uploaded as plain integer and double arrays and read by a
 * fixed kernel, so a new tape costs an upload rather than an NVRTC invocation.
 * The generated kernel it is measured against needs seconds of compilation
 * before its first scenario; this one is ready immediately, which is what
 * matters when the tape shape changes as often as it is replayed.
 *
 * <p>Node values cannot live in registers here, because the node number is a
 * runtime index rather than a literal. They live instead in a scratch buffer
 * laid out as {@code [node][thread]}, so the threads of a warp — which are all
 * evaluating the same node at the same moment — touch consecutive addresses and
 * coalesce into full cache lines. That layout is what keeps an interpreter
 * within single digits of the generated kernel instead of an order of magnitude
 * behind it, and it makes the engine bandwidth bound: the scratch traffic per
 * scenario is roughly the tape size times the width of a value, twice over for
 * the adjoint sweep.
 */
final class CudaInterpExecutable extends AbstractAadExecutable {

  private static final int BLOCK = CudaTapeKernels.BLOCK;
  private static final int BLOCKS = Integer.getInteger("nablatensor.interp.blocks", 32);

  private final long function;
  private final int channels;
  private final double compileSeconds;
  private final int threads;

  private final long nodesBuffer;
  private final long constsBuffer;
  private final long inputsBuffer;
  private final long inputNodesBuffer;
  private final long metaBuffer;
  private final long valuesBuffer;
  private final long adjointsBuffer;
  private final long partialsBuffer;

  CudaInterpExecutable(AadTape tape, AadOptions options) {
    super(tape, options);
    if (tape.inputCount() > CudaTapeKernels.MAX_INPUTS) {
      throw new IllegalArgumentException("the interpreting engine supports at most "
          + CudaTapeKernels.MAX_INPUTS + " inputs");
    }
    CudaTapeKernels.Handles handles = CudaTapeKernels.get(options.precision());
    this.function = handles.interp();
    this.compileSeconds = handles.compileSeconds();
    this.channels = options.adjoints() ? tape.inputCount() + 1 : 1;
    this.threads = BLOCKS * BLOCK;

    int n = tape.size();
    int[] nodes = new int[n * 4];
    double[] constants = new double[n];
    for (int i = 0; i < n; i++) {
      nodes[i * 4] = CudaTapeKernels.opcode(tape.op(i));
      nodes[i * 4 + 1] = tape.argA(i);
      nodes[i * 4 + 2] = tape.argB(i);
      nodes[i * 4 + 3] = tape.isActive(i) ? 1 : 0;
      constants[i] = tape.constant(i);
    }
    int[] inputNodes = new int[tape.inputCount()];
    for (int j = 0; j < inputNodes.length; j++) {
      inputNodes[j] = tape.inputNode(j);
    }
    int[] meta = {n, tape.inputCount(), tape.outputNode(), options.adjoints() ? 1 : 0};

    this.nodesBuffer = CudaJit.malloc((long) nodes.length * Integer.BYTES);
    this.constsBuffer = CudaJit.malloc((long) Math.max(1, n) * Double.BYTES);
    this.inputsBuffer = CudaJit.malloc((long) Math.max(1, inputs.length) * Double.BYTES);
    this.inputNodesBuffer = CudaJit.malloc((long) Math.max(1, inputNodes.length) * Integer.BYTES);
    this.metaBuffer = CudaJit.malloc((long) meta.length * Integer.BYTES);
    CudaJit.uploadInts(nodesBuffer, nodes);
    CudaJit.uploadDoubles(constsBuffer, constants);
    CudaJit.uploadInts(inputNodesBuffer, inputNodes.length == 0 ? new int[1] : inputNodes);
    CudaJit.uploadInts(metaBuffer, meta);

    long elementBytes = options.precision() == AadOptions.Precision.FLOAT32 ? 4 : 8;
    long scratch = (long) n * threads * elementBytes;
    this.valuesBuffer = CudaJit.malloc(scratch);
    this.adjointsBuffer = options.adjoints() ? CudaJit.malloc(scratch) : CudaJit.malloc(elementBytes);
    this.partialsBuffer = CudaJit.malloc((long) BLOCKS * channels * Double.BYTES);
  }

  @Override
  public String engineName() {
    return "cuda-interp";
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

  @Override
  public AadResult replay(long paths, long pathOffset, long seed) {
    checkOpen();
    if (paths <= 0) {
      throw new IllegalArgumentException("paths must be positive");
    }
    CudaJit.uploadDoubles(inputsBuffer, inputs);

    long start = System.nanoTime();
    CudaJit.launch(function, BLOCKS, BLOCK,
        nodesBuffer, constsBuffer, inputsBuffer, inputNodesBuffer, metaBuffer,
        paths, pathOffset, seed, valuesBuffer, adjointsBuffer, partialsBuffer);
    CudaJit.synchronize();
    double seconds = (System.nanoTime() - start) / 1e9;

    double[] partials = CudaJit.downloadDoubles(partialsBuffer, BLOCKS * channels);
    double value = 0.0;
    double[] gradients = new double[tape.inputCount()];
    for (int block = 0; block < BLOCKS; block++) {
      value += partials[block * channels];
      for (int c = 1; c < channels; c++) {
        gradients[c - 1] += partials[block * channels + c];
      }
    }
    value /= paths;
    for (int j = 0; j < gradients.length; j++) {
      gradients[j] /= paths;
    }
    return new AadResult(value, gradients, tape.inputNames(), paths, seconds);
  }

  @Override
  public void close() {
    super.close();
    for (long buffer : new long[] {nodesBuffer, constsBuffer, inputsBuffer, inputNodesBuffer,
        metaBuffer, valuesBuffer, adjointsBuffer, partialsBuffer}) {
      if (buffer != 0) {
        CudaJit.free(buffer);
      }
    }
  }
}
