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
package com.nablatensor.backend.rocm;

import com.nablatensor.tensor.ConvSpec;
import com.nablatensor.tensor.DType;
import com.nablatensor.tensor.Device;
import com.nablatensor.tensor.DeviceType;
import com.nablatensor.tensor.Op;
import com.nablatensor.tensor.Shape;
import com.nablatensor.tensor.expr.Expr;
import com.nablatensor.tensor.spi.ComputeBackend;
import com.nablatensor.tensor.spi.DeviceBuffer;
import com.nablatensor.tensor.spi.GpuKernel;
import com.nablatensor.tensor.spi.GpuKernels;

import java.util.HashMap;
import java.util.Map;

/**
 * AMD ROCm/HIP backend. Math runs in custom kernels compiled to a GCN code
 * object at runtime with HIPRTC and launched through the HIP runtime via
 * {@link java.lang.foreign}; tensors stay resident in device memory between
 * operations. The kernel source is {@link GpuKernels}, shared verbatim with the
 * CUDA backend — HIPRTC accepts the CUDA C unchanged.
 *
 * <p>Phase-7 first cut: elementwise / reductions / matmul / transpose /
 * conv2d / broadcast are implemented.
 */
public final class RocmBackend implements ComputeBackend {

  private static final int BLOCK = 256;
  private static final GpuKernel MATMUL = GpuKernels.kernel("matmul_tiled");
  private static final GpuKernel BATCHED_MATMUL = GpuKernels.kernel("batched_matmul_tiled");

  private static final int MAX_BROADCAST_RANK = 4;

  private boolean initialized;
  private HipRuntime.DeviceInfo deviceInfo;
  private final Map<String, Long> functions = new HashMap<>();
  private final Map<String, Long> fusedFunctions = new HashMap<>();

  @Override
  public String name() {
    return "rocm";
  }

  @Override
  public DeviceType deviceType() {
    return DeviceType.ROCM;
  }

  @Override
  public boolean isAvailable() {
    try {
      return HipRuntime.probe();
    } catch (Throwable failure) {
      return false;
    }
  }

  @Override
  public int priority() {
    return 50;
  }

  public String deviceName() {
    ensureInitialized();
    return deviceInfo.name();
  }

  private synchronized void ensureInitialized() {
    if (initialized) {
      return;
    }
    deviceInfo = HipRuntime.context();
    byte[] code = HipRuntime.compile(GpuKernels.TENSOR_SOURCE, deviceInfo.arch());
    for (String kernel : GpuKernels.TENSOR_KERNEL_NAMES) {
      functions.put(kernel, HipRuntime.loadFunction(code, kernel));
    }
    initialized = true;
  }

  // ---- data movement ---------------------------------------------------------

  @Override
  public DeviceBuffer upload(float[] data, Shape shape, DType dtype, Device device) {
    ensureInitialized();
    requireRocmDevice(device);
    if (dtype != DType.F32) {
      throw new UnsupportedOperationException("ROCm float upload supports F32 only, got " + dtype);
    }
    HipRuntime.synchronize();
    return new HipBuffer(HipRuntime.uploadFloats(data), shape, dtype, device);
  }

  @Override
  public float[] download(DeviceBuffer buffer) {
    ensureInitialized();
    HipRuntime.synchronize();
    HipBuffer hip = hip(buffer);
    return HipRuntime.downloadFloats(hip.pointer, hip.count());
  }

  @Override
  public DeviceBuffer randomUniform(long seed, long counter, Shape shape, Device device) {
    ensureInitialized();
    requireRocmDevice(device);
    int size = Math.toIntExact(shape.size());
    HipBuffer out = alloc(shape);
    HipRuntime.launch(functions.get("random_uniform"), grid(size), BLOCK, out.pointer, seed, counter, size);
    return out;
  }

  @Override
  public DeviceBuffer randomNormal(long seed, long counter, Shape shape, Device device) {
    ensureInitialized();
    requireRocmDevice(device);
    int size = Math.toIntExact(shape.size());
    HipBuffer out = alloc(shape);
    HipRuntime.launch(functions.get("random_normal"), grid(size), BLOCK, out.pointer, seed, counter, size);
    return out;
  }

  // ---- elementwise ---------------------------------------------------------

  @Override
  public DeviceBuffer binary(Op op, DeviceBuffer a, DeviceBuffer b) {
    ensureInitialized();
    HipBuffer left = hip(a);
    HipBuffer right = hip(b);
    int n = left.count();
    if (n != right.count()) {
      throw new IllegalArgumentException("shape mismatch: " + left.shape() + " vs " + right.shape());
    }
    HipBuffer out = alloc(left.shape());
    HipRuntime.launch(functions.get("ew_binary"), grid(n), BLOCK,
        out.pointer, left.pointer, right.pointer, n, binaryCode(op));
    return out;
  }

  @Override
  public DeviceBuffer scalar(Op op, DeviceBuffer a, double value) {
    ensureInitialized();
    HipBuffer left = hip(a);
    int n = left.count();
    HipBuffer out = alloc(left.shape());
    HipRuntime.launch(functions.get("ew_scalar"), grid(n), BLOCK,
        out.pointer, left.pointer, (float) value, n, binaryCode(op));
    return out;
  }

  @Override
  public DeviceBuffer unary(Op op, DeviceBuffer a) {
    ensureInitialized();
    HipBuffer left = hip(a);
    int n = left.count();
    HipBuffer out = alloc(left.shape());
    HipRuntime.launch(functions.get("ew_unary"), grid(n), BLOCK, out.pointer, left.pointer, n, unaryCode(op));
    return out;
  }

  // ---- linear algebra ----------------------------------------------------

  @Override
  public DeviceBuffer transpose(DeviceBuffer a) {
    ensureInitialized();
    HipBuffer left = hip(a);
    Shape shape = left.shape();
    if (shape.rank() != 2) {
      throw new IllegalStateException("transpose requires a rank-2 tensor, got " + shape);
    }
    int rows = shape.dim(0);
    int cols = shape.dim(1);
    HipBuffer out = alloc(shape.transposed());
    HipRuntime.launch(functions.get("transpose2d"), grid(rows * cols), BLOCK, out.pointer, left.pointer, rows, cols);
    return out;
  }

  @Override
  public DeviceBuffer matmul(DeviceBuffer a, DeviceBuffer b) {
    ensureInitialized();
    HipBuffer left = hip(a);
    HipBuffer right = hip(b);
    Shape ls = left.shape();
    Shape rs = right.shape();
    requireMatchingPlacement(left, right);
    if (ls.rank() != 2 || rs.rank() != 2 || ls.dim(1) != rs.dim(0)) {
      throw new IllegalArgumentException("incompatible matmul shapes: " + ls + " x " + rs);
    }
    int m = ls.dim(0);
    int k = ls.dim(1);
    int n = rs.dim(1);
    requirePositiveMatmulDimensions(m, k, n);
    HipBuffer out = alloc(Shape.of(m, n));
    long totalTiles = Math.multiplyExact((long) gridTiles(m), gridTiles(n));
    for (long offset = 0; offset < totalTiles; offset += Integer.MAX_VALUE) {
      int blocks = Math.toIntExact(Math.min(Integer.MAX_VALUE, totalTiles - offset));
      HipRuntime.launchBlocks2d(functions.get(MATMUL.name()), blocks,
          MATMUL.blockDimX(), MATMUL.blockDimY(),
          out.pointer, left.pointer, right.pointer, m, k, n, offset);
    }
    return out;
  }

  @Override
  public DeviceBuffer batchedMatmul(DeviceBuffer a, DeviceBuffer b) {
    ensureInitialized();
    HipBuffer left = hip(a);
    HipBuffer right = hip(b);
    Shape ls = left.shape();
    Shape rs = right.shape();
    requireMatchingPlacement(left, right);
    if (ls.rank() != 3 || rs.rank() != 3 || ls.dim(0) != rs.dim(0) || ls.dim(2) != rs.dim(1)) {
      throw new IllegalArgumentException("incompatible batched matmul shapes: " + ls + " x " + rs);
    }
    int batch = ls.dim(0);
    int m = ls.dim(1);
    int k = ls.dim(2);
    int n = rs.dim(2);
    requirePositiveMatmulDimensions(batch, m, k, n);
    HipBuffer out = alloc(Shape.of(batch, m, n));
    long tilesPerBatch = Math.multiplyExact((long) gridTiles(m), gridTiles(n));
    long totalTiles = Math.multiplyExact((long) batch, tilesPerBatch);
    for (long offset = 0; offset < totalTiles; offset += Integer.MAX_VALUE) {
      int blocks = Math.toIntExact(Math.min(Integer.MAX_VALUE, totalTiles - offset));
      HipRuntime.launchBlocks2d(functions.get(BATCHED_MATMUL.name()), blocks,
          BATCHED_MATMUL.blockDimX(), BATCHED_MATMUL.blockDimY(),
          out.pointer, left.pointer, right.pointer, batch, m, k, n, offset);
    }
    return out;
  }

  // ---- reductions ------------------------------------------------------------

  @Override
  public DeviceBuffer reduceSum(DeviceBuffer a) {
    ensureInitialized();
    HipBuffer in = hip(a);
    HipBuffer out = alloc(Shape.of(1));
    HipRuntime.launch(functions.get("reduce_sum"), 1, BLOCK, out.pointer, in.pointer, in.count());
    return out;
  }

  @Override
  public DeviceBuffer reduceMax(DeviceBuffer a) {
    ensureInitialized();
    HipBuffer in = hip(a);
    HipBuffer out = alloc(Shape.of(1));
    HipRuntime.launch(functions.get("reduce_max"), 1, BLOCK, out.pointer, in.pointer, in.count());
    return out;
  }

  @Override
  public DeviceBuffer sumAxis0(DeviceBuffer a) {
    ensureInitialized();
    Shape shape = a.shape();
    if (shape.rank() != 2) {
      throw new IllegalStateException("sumAxis0 requires a rank-2 tensor, got " + shape);
    }
    return sumAxis(a, 0, false);
  }

  @Override
  public DeviceBuffer sumAxis(DeviceBuffer a, int axis, boolean keepDims) {
    return reduceAxis(a, axis, keepDims, "reduce_axis_sum");
  }

  @Override
  public DeviceBuffer maxAxis(DeviceBuffer a, int axis, boolean keepDims) {
    return reduceAxis(a, axis, keepDims, "reduce_axis_max");
  }

  @Override
  public DeviceBuffer argMaxAxis(DeviceBuffer a, int axis) {
    return reduceAxis(a, axis, false, "reduce_axis_argmax");
  }

  @Override
  public DeviceBuffer maxAxisBackward(DeviceBuffer upstream, DeviceBuffer input, int axis) {
    ensureInitialized();
    HipBuffer grad = hip(upstream);
    HipBuffer in = hip(input);
    Shape shape = in.shape();
    if (axis < 0 || axis >= shape.rank()) {
      throw new IllegalArgumentException("axis " + axis + " is out of bounds for shape " + shape);
    }
    int outer = 1;
    for (int i = 0; i < axis; i++) {
      outer = Math.multiplyExact(outer, shape.dim(i));
    }
    int inner = 1;
    for (int i = axis + 1; i < shape.rank(); i++) {
      inner = Math.multiplyExact(inner, shape.dim(i));
    }
    if (grad.count() != outer * inner) {
      throw new IllegalArgumentException("maxAxisBackward gradient shape " + grad.shape()
          + " is incompatible with input " + shape + " and axis " + axis);
    }
    HipBuffer out = alloc(shape);
    HipRuntime.launch(functions.get("reduce_axis_max_backward"), grid(outer * inner), BLOCK,
        out.pointer, grad.pointer, in.pointer, outer, shape.dim(axis), inner);
    return out;
  }

  private DeviceBuffer reduceAxis(DeviceBuffer buffer, int axis, boolean keepDims, String kernel) {
    ensureInitialized();
    HipBuffer input = hip(buffer);
    Shape shape = input.shape();
    if (axis < 0 || axis >= shape.rank()) {
      throw new IllegalArgumentException("axis " + axis + " is out of bounds for shape " + shape);
    }
    int outer = 1;
    for (int i = 0; i < axis; i++) {
      outer = Math.multiplyExact(outer, shape.dim(i));
    }
    int inner = 1;
    for (int i = axis + 1; i < shape.rank(); i++) {
      inner = Math.multiplyExact(inner, shape.dim(i));
    }
    int[] outputDims;
    if (keepDims) {
      outputDims = shape.dims();
      outputDims[axis] = 1;
    } else {
      int[] dims = shape.dims();
      outputDims = new int[dims.length - 1];
      System.arraycopy(dims, 0, outputDims, 0, axis);
      System.arraycopy(dims, axis + 1, outputDims, axis, dims.length - axis - 1);
    }
    HipBuffer out = alloc(Shape.of(outputDims));
    int outputSize = outer * inner;
    HipRuntime.launch(functions.get(kernel), grid(outputSize), BLOCK,
        out.pointer, input.pointer, outer, shape.dim(axis), inner);
    return out;
  }

  // ---- convolution -------------------------------------------------------

  @Override
  public DeviceBuffer conv2d(DeviceBuffer x, DeviceBuffer w, ConvSpec spec) {
    ensureInitialized();
    HipBuffer in = hip(x);
    HipBuffer filters = hip(w);
    int batch = in.shape().dim(0);
    HipBuffer out = alloc(spec.outputShape(batch));
    int total = batch * spec.outputSize();
    HipRuntime.launch(functions.get("conv2d_fwd"), grid(total), BLOCK,
        out.pointer, in.pointer, filters.pointer,
        batch, spec.inChannels(), spec.inHeight(), spec.inWidth(),
        spec.outChannels(), spec.kernel(), spec.stride(), spec.pad(),
        spec.outHeight(), spec.outWidth());
    return out;
  }

  @Override
  public DeviceBuffer conv2dGradInput(DeviceBuffer upstream, DeviceBuffer w, ConvSpec spec) {
    ensureInitialized();
    HipBuffer grad = hip(upstream);
    HipBuffer filters = hip(w);
    int batch = grad.shape().dim(0);
    HipBuffer out = alloc(spec.inputShape(batch));
    int total = batch * spec.inputSize();
    HipRuntime.launch(functions.get("conv2d_dx"), grid(total), BLOCK,
        out.pointer, grad.pointer, filters.pointer,
        batch, spec.inChannels(), spec.inHeight(), spec.inWidth(),
        spec.outChannels(), spec.kernel(), spec.stride(), spec.pad(),
        spec.outHeight(), spec.outWidth());
    return out;
  }

  @Override
  public DeviceBuffer conv2dGradWeight(DeviceBuffer x, DeviceBuffer upstream, ConvSpec spec) {
    ensureInitialized();
    HipBuffer in = hip(x);
    HipBuffer grad = hip(upstream);
    int batch = in.shape().dim(0);
    HipBuffer out = alloc(spec.weightShape());
    int total = spec.outChannels() * spec.weightSize();
    HipRuntime.launch(functions.get("conv2d_dw"), grid(total), BLOCK,
        out.pointer, in.pointer, grad.pointer,
        batch, spec.inChannels(), spec.inHeight(), spec.inWidth(),
        spec.outChannels(), spec.kernel(), spec.stride(), spec.pad(),
        spec.outHeight(), spec.outWidth());
    return out;
  }

  @Override
  public DeviceBuffer reluBackward(DeviceBuffer upstream, DeviceBuffer input) {
    ensureInitialized();
    HipBuffer grad = hip(upstream);
    HipBuffer in = hip(input);
    int n = grad.count();
    HipBuffer out = alloc(grad.shape());
    HipRuntime.launch(functions.get("relu_backward"), grid(n), BLOCK, out.pointer, grad.pointer, in.pointer, n);
    return out;
  }

  // ---- fused elementwise --------------------------------------------------

  /**
   * Runs a whole elementwise expression chain as one generated HIP kernel
   * launch, instead of one launch (and one intermediate device buffer) per op.
   * The generated source is CUDA-syntax, which HIPRTC accepts unchanged; the
   * compiled kernel is cached by its generated source, so repeated fused calls
   * for the same op chain (any shape) reuse it without recompiling.
   */
  @Override
  public DeviceBuffer fused(Expr expr, DeviceBuffer[] inputs) {
    ensureInitialized();
    HipBuffer[] buffers = new HipBuffer[inputs.length];
    int n = -1;
    for (int i = 0; i < inputs.length; i++) {
      buffers[i] = hip(inputs[i]);
      if (i == 0) {
        n = buffers[i].count();
      } else if (buffers[i].count() != n) {
        throw new IllegalArgumentException("fused inputs must share element count");
      }
    }
    String source = GpuKernels.fusedSource(expr, inputs.length);
    long function = fusedFunctions.computeIfAbsent(source,
        src -> HipRuntime.loadFunction(
            HipRuntime.compile(src, deviceInfo.arch()), GpuKernels.FUSED_KERNEL_NAME));
    HipBuffer out = alloc(buffers[0].shape());
    Object[] arguments = new Object[inputs.length + 2];
    arguments[0] = out.pointer;
    for (int i = 0; i < inputs.length; i++) {
      arguments[i + 1] = buffers[i].pointer;
    }
    arguments[inputs.length + 1] = n;
    HipRuntime.launch(function, grid(n), BLOCK, arguments);
    return out;
  }

  // ---- shape ops -------------------------------------------------------------

  @Override
  public DeviceBuffer reshape(DeviceBuffer a, Shape target) {
    ensureInitialized();
    HipBuffer input = hip(a);
    HipBuffer out = alloc(target);
    HipRuntime.copyDevice(out.pointer, input.pointer, target.size() * input.dtype().byteSize());
    return out;
  }

  @Override
  public DeviceBuffer broadcastTo(DeviceBuffer a, Shape target) {
    ensureInitialized();
    Shape src = a.shape();
    if (src.equals(target)) {
      return a;
    }
    int rank = target.rank();
    if (rank > MAX_BROADCAST_RANK) {
      throw new UnsupportedOperationException("broadcastTo supports up to rank " + MAX_BROADCAST_RANK + ", got " + target);
    }
    int pad = rank - src.rank();
    int[] srcDims = new int[rank];
    for (int i = 0; i < rank; i++) {
      srcDims[i] = i < pad ? 1 : src.dim(i - pad);
    }
    int[] srcStrides = new int[rank];
    int stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
      srcStrides[i] = srcDims[i] == 1 ? 0 : stride;
      stride *= srcDims[i];
    }
    int[] targetDims = target.dims();
    int leadPad = MAX_BROADCAST_RANK - rank;
    int[] d = new int[MAX_BROADCAST_RANK];
    int[] s = new int[MAX_BROADCAST_RANK];
    for (int i = 0; i < MAX_BROADCAST_RANK; i++) {
      d[i] = i < leadPad ? 1 : targetDims[i - leadPad];
      s[i] = i < leadPad ? 0 : srcStrides[i - leadPad];
    }
    HipBuffer in = hip(a);
    HipBuffer out = alloc(target);
    int n = Math.toIntExact(target.size());
    HipRuntime.launch(functions.get("broadcast_to"), grid(n), BLOCK,
        out.pointer, in.pointer, d[0], d[1], d[2], d[3], s[0], s[1], s[2], s[3], n);
    return out;
  }

  @Override
  public DeviceBuffer sliceAxis0(DeviceBuffer input, int index) {
    ensureInitialized();
    HipBuffer in = hip(input);
    Shape shape = in.shape();
    if (shape.rank() < 1 || index < 0 || index >= shape.dim(0)) {
      throw new IllegalArgumentException("axis-0 index " + index + " is out of bounds for " + shape);
    }
    long sliceSize = shape.size() / shape.dim(0);
    int[] dims = shape.dims();
    int[] outputDims = new int[dims.length - 1];
    System.arraycopy(dims, 1, outputDims, 0, outputDims.length);
    HipBuffer out = alloc(Shape.of(outputDims), in.dtype(), in.device());
    long bytes = Math.multiplyExact(sliceSize, in.dtype().byteSize());
    long sourceOffset = Math.multiplyExact(Math.multiplyExact((long) index, sliceSize), in.dtype().byteSize());
    HipRuntime.copyDevice(out.pointer, Math.addExact(in.pointer, sourceOffset), bytes);
    return out;
  }

  @Override
  public DeviceBuffer stackAxis0(DeviceBuffer[] inputs) {
    ensureInitialized();
    if (inputs.length == 0) {
      throw new IllegalArgumentException("stackAxis0 requires at least one input");
    }
    HipBuffer first = hip(inputs[0]);
    long sliceSize = first.shape().size();
    int[] inputDims = first.shape().dims();
    int[] outputDims = new int[inputDims.length + 1];
    outputDims[0] = inputs.length;
    System.arraycopy(inputDims, 0, outputDims, 1, inputDims.length);
    HipBuffer out = alloc(Shape.of(outputDims), first.dtype(), first.device());
    long bytes = Math.multiplyExact(sliceSize, first.dtype().byteSize());
    for (int i = 0; i < inputs.length; i++) {
      HipBuffer input = hip(inputs[i]);
      if (!input.shape().equals(first.shape()) || input.dtype() != first.dtype()
          || !input.device().equals(first.device())) {
        release(out);
        throw new IllegalArgumentException("stackAxis0 inputs must have identical shape, dtype, and device");
      }
      long destinationOffset = Math.multiplyExact(Math.multiplyExact((long) i, sliceSize), first.dtype().byteSize());
      HipRuntime.copyDevice(Math.addExact(out.pointer, destinationOffset), input.pointer, bytes);
    }
    return out;
  }

  // ---- lifecycle ----------------------------------------------------------

  @Override
  public void sync() {
    if (initialized) {
      HipRuntime.synchronize();
    }
  }

  @Override
  public void release(DeviceBuffer buffer) {
    hip(buffer).release();
  }

  // ---- helpers -----------------------------------------------------------

  private HipBuffer alloc(Shape shape) {
    return alloc(shape, DType.F32, Device.rocm());
  }

  private HipBuffer alloc(Shape shape, DType dtype, Device device) {
    requireRocmDevice(device);
    long bytes = Math.multiplyExact(shape.size(), dtype.byteSize());
    return new HipBuffer(HipRuntime.malloc(bytes), shape, dtype, device);
  }

  private static int grid(int n) {
    return n == 0 ? 0 : 1 + (n - 1) / BLOCK;
  }

  private static int gridTiles(int size) {
    return 1 + (size - 1) / MATMUL.blockDimX();
  }

  private static void requirePositiveMatmulDimensions(int... dimensions) {
    for (int dimension : dimensions) {
      if (dimension < 1) {
        throw new IllegalArgumentException("matmul dimensions must be positive");
      }
    }
  }

  private static void requireRocmDevice(Device device) {
    if (!device.equals(Device.rocm())) {
      throw new IllegalArgumentException("ROCm backend currently supports only rocm:0, got " + device);
    }
  }

  private static void requireMatchingPlacement(HipBuffer left, HipBuffer right) {
    if (left.dtype() != right.dtype() || !left.device().equals(right.device())) {
      throw new IllegalArgumentException("matmul operands must use the same device and dtype");
    }
    requireRocmDevice(left.device());
    if (left.dtype() != DType.F32) {
      throw new UnsupportedOperationException("ROCm matmul supports F32 only, got " + left.dtype());
    }
  }

  private static HipBuffer hip(DeviceBuffer buffer) {
    if (buffer instanceof HipBuffer hip) {
      return hip;
    }
    throw new IllegalArgumentException("expected a ROCm buffer, got " + buffer.getClass());
  }

  private static int binaryCode(Op op) {
    return switch (op) {
      case ADD -> 0;
      case SUB -> 1;
      case MUL -> 2;
      case DIV -> 3;
      case MAX -> 4;
      case MIN -> 5;
      default -> throw new IllegalArgumentException("not a binary/scalar op: " + op);
    };
  }

  private static int unaryCode(Op op) {
    return switch (op) {
      case NEG -> 0;
      case EXP -> 1;
      case LOG -> 2;
      case SQRT -> 3;
      case RSQRT -> 4;
      case TANH -> 5;
      case SIGMOID -> 6;
      case RELU -> 7;
      case ABS -> 8;
      case SIGN -> 9;
      default -> throw new IllegalArgumentException("not a unary op: " + op);
    };
  }
}
