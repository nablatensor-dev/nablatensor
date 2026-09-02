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
package com.nablatensor.tensor.spi;

import com.nablatensor.tensor.ConvSpec;
import com.nablatensor.tensor.DType;
import com.nablatensor.tensor.Device;
import com.nablatensor.tensor.DeviceType;
import com.nablatensor.tensor.Op;
import com.nablatensor.tensor.Shape;
import com.nablatensor.tensor.expr.Expr;

import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * A compute backend. Implementations are discovered with {@link java.util.ServiceLoader}
 * and must be registered in
 * {@code META-INF/services/com.nablatensor.tensor.spi.ComputeBackend}.
 *
 * <p>Every numerical operation is executed by a custom kernel owned by the
 * backend (runtime-compiled CUDA/ROCm, or a generated/vectorized CPU kernel).
 */
public interface ComputeBackend {

  String name();

  DeviceType deviceType();

  /** Whether this backend can run on the current machine (drivers/GPU present). */
  boolean isAvailable();

  /** Higher wins when {@code Backend.AUTO} selects a default (CUDA &gt; Vulkan &gt; ROCm &gt; CPU). */
  int priority();

  DeviceBuffer upload(float[] data, Shape shape, DType dtype, Device device);

  /** Deterministic counter-based uniform values in {@code [0, 1)}. */
  default DeviceBuffer randomUniform(long seed, long counter, Shape shape, Device device) {
    int size = Math.toIntExact(shape.size());
    float[] values = new float[size];
    for (int i = 0; i < size; i++) {
      values[i] = uniform(seed, counter + i);
    }
    return upload(values, shape, DType.F32, device);
  }

  /** Deterministic counter-based standard-normal values generated with Box-Muller. */
  default DeviceBuffer randomNormal(long seed, long counter, Shape shape, Device device) {
    int size = Math.toIntExact(shape.size());
    float[] values = new float[size];
    for (int i = 0; i < size; i++) {
      float u1 = uniformOpen(seed, counter + 2L * i);
      float u2 = uniform(seed, counter + 2L * i + 1);
      values[i] = (float) (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2));
    }
    return upload(values, shape, DType.F32, device);
  }

  float[] download(DeviceBuffer buffer);

  DeviceBuffer binary(Op op, DeviceBuffer a, DeviceBuffer b);

  DeviceBuffer scalar(Op op, DeviceBuffer a, double value);

  DeviceBuffer unary(Op op, DeviceBuffer a);

  DeviceBuffer transpose(DeviceBuffer a);

  DeviceBuffer matmul(DeviceBuffer a, DeviceBuffer b);

  default DeviceBuffer batchedMatmul(DeviceBuffer a, DeviceBuffer b) {
    throw new UnsupportedOperationException(name() + " backend does not support batchedMatmul");
  }

  /** Copies one slice from axis zero, removing that axis from the result shape. */
  default DeviceBuffer sliceAxis0(DeviceBuffer input, int index) {
    Shape shape = input.shape();
    if (shape.rank() < 1 || index < 0 || index >= shape.dim(0)) {
      throw new IllegalArgumentException("axis-0 index " + index + " is out of bounds for " + shape);
    }
    int sliceSize = Math.toIntExact(shape.size() / shape.dim(0));
    float[] values = download(input);
    float[] slice = Arrays.copyOfRange(values, index * sliceSize, (index + 1) * sliceSize);
    int[] dims = Arrays.copyOfRange(shape.dims(), 1, shape.rank());
    return upload(slice, Shape.of(dims), input.dtype(), input.device());
  }

  /** Stacks equally-shaped buffers along a new leading axis. */
  default DeviceBuffer stackAxis0(DeviceBuffer[] inputs) {
    if (inputs.length == 0) {
      throw new IllegalArgumentException("stackAxis0 requires at least one input");
    }
    DeviceBuffer first = inputs[0];
    int sliceSize = Math.toIntExact(first.shape().size());
    float[] values = new float[Math.multiplyExact(inputs.length, sliceSize)];
    for (int i = 0; i < inputs.length; i++) {
      DeviceBuffer input = inputs[i];
      if (!input.shape().equals(first.shape())
          || input.dtype() != first.dtype()
          || !input.device().equals(first.device())) {
        throw new IllegalArgumentException("stackAxis0 inputs must have identical shape, dtype, and device");
      }
      float[] slice = download(input);
      System.arraycopy(slice, 0, values, i * sliceSize, sliceSize);
    }
    int[] inputDims = first.shape().dims();
    int[] outputDims = new int[inputDims.length + 1];
    outputDims[0] = inputs.length;
    System.arraycopy(inputDims, 0, outputDims, 1, inputDims.length);
    return upload(values, Shape.of(outputDims), first.dtype(), first.device());
  }

  /** Sums every element, returning a single-element buffer of shape {@code (1)}. */
  DeviceBuffer reduceSum(DeviceBuffer a);

  /** Maximum over every element, returning a single-element buffer of shape {@code (1)}. */
  DeviceBuffer reduceMax(DeviceBuffer a);

  /** Sums a rank-2 buffer over axis 0, returning a rank-1 buffer of shape {@code (cols)}. */
  DeviceBuffer sumAxis0(DeviceBuffer a);

  default DeviceBuffer sumAxis(DeviceBuffer a, int axis, boolean keepDims) {
    throw new UnsupportedOperationException(name() + " backend does not support sumAxis");
  }

  default DeviceBuffer maxAxis(DeviceBuffer a, int axis, boolean keepDims) {
    throw new UnsupportedOperationException(name() + " backend does not support maxAxis");
  }

  /**
   * Returns axis indices in an F32 buffer. Indices above 2^24 cannot be represented exactly
   * until integer tensor dtypes are supported.
   */
  default DeviceBuffer argMaxAxis(DeviceBuffer a, int axis) {
    throw new UnsupportedOperationException(name() + " backend does not support argMaxAxis");
  }

  /**
   * Routes one upstream value per reduced slice to the first maximum input position.
   */
  default DeviceBuffer maxAxisBackward(DeviceBuffer upstream, DeviceBuffer input, int axis) {
    throw new UnsupportedOperationException(name() + " backend does not support maxAxisBackward");
  }

  /**
   * Copies a buffer while changing only its shape. Backends may override this to keep the copy
   * device-local.
   */
  default DeviceBuffer reshape(DeviceBuffer a, Shape target) {
    return upload(download(a), target, a.dtype(), a.device());
  }

  /**
   * 2-D convolution, no bias. {@code x} is {@code (batch, inC*inH*inW)},
   * {@code w} is {@code (outC, inC*k*k)}, result is {@code (batch, outC*outH*outW)}.
   */
  DeviceBuffer conv2d(DeviceBuffer x, DeviceBuffer w, ConvSpec spec);

  /** Gradient of {@link #conv2d} with respect to its input. */
  DeviceBuffer conv2dGradInput(DeviceBuffer upstream, DeviceBuffer w, ConvSpec spec);

  /** Gradient of {@link #conv2d} with respect to its weights. */
  DeviceBuffer conv2dGradWeight(DeviceBuffer x, DeviceBuffer upstream, ConvSpec spec);

  /** Routes {@code upstream} through a ReLU: {@code input > 0 ? upstream : 0}. */
  DeviceBuffer reluBackward(DeviceBuffer upstream, DeviceBuffer input);

  /** Broadcasts {@code a} to {@code target} following {@link Shape#broadcast}'s rules. */
  DeviceBuffer broadcastTo(DeviceBuffer a, Shape target);

  /**
   * Executes a fused elementwise expression tree ({@code expr}, reading from
   * {@code inputs}) as a single unit. The default walks the tree with the
   * primitive {@link #unary}/{@link #binary}/{@link #scalar} ops (no real
   * fusion, one call per node); CUDA/CPU backends override this to run the
   * whole chain as a single kernel launch / pass.
   */
  default DeviceBuffer fused(Expr expr, DeviceBuffer[] inputs) {
    return evalFusedDefault(expr, inputs, new IdentityHashMap<>());
  }

  private DeviceBuffer evalFusedDefault(Expr expr, DeviceBuffer[] inputs, Map<Expr, DeviceBuffer> memo) {
    DeviceBuffer cached = memo.get(expr);
    if (cached != null) {
      return cached;
    }
    DeviceBuffer result = switch (expr) {
      case Expr.Input in -> inputs[in.index()];
      case Expr.Unary u -> unary(u.op(), evalFusedDefault(u.in(), inputs, memo));
      case Expr.Binary b ->
          binary(b.op(), evalFusedDefault(b.left(), inputs, memo), evalFusedDefault(b.right(), inputs, memo));
      case Expr.Scalar s -> scalar(s.op(), evalFusedDefault(s.in(), inputs, memo), s.value());
    };
    memo.put(expr, result);
    return result;
  }

  /** Block until all queued kernels on this backend have completed. */
  default void sync() {
  }

  /** Deterministically frees a buffer now, instead of waiting on GC/Cleaner timing. */
  default void release(DeviceBuffer buffer) {
  }

  private static float uniform(long seed, long counter) {
    return (mix64(seed + 0x9E3779B97F4A7C15L * counter) >>> 40) * 0x1.0p-24f;
  }

  private static float uniformOpen(long seed, long counter) {
    return ((mix64(seed + 0x9E3779B97F4A7C15L * counter) >>> 40) + 1) * 0x1.0p-24f;
  }

  private static long mix64(long value) {
    long mixed = value;
    mixed = (mixed ^ (mixed >>> 30)) * 0xBF58476D1CE4E5B9L;
    mixed = (mixed ^ (mixed >>> 27)) * 0x94D049BB133111EBL;
    return mixed ^ (mixed >>> 31);
  }
}
