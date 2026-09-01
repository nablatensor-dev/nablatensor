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
package com.nablatensor.backend.cuda;

import com.nablatensor.tensor.ConvSpec;
import com.nablatensor.tensor.DType;
import com.nablatensor.tensor.Device;
import com.nablatensor.tensor.DeviceType;
import com.nablatensor.tensor.Op;
import com.nablatensor.tensor.Shape;
import com.nablatensor.tensor.expr.Expr;
import com.nablatensor.tensor.spi.ComputeBackend;
import com.nablatensor.tensor.spi.DeviceBuffer;
import com.nablatensor.tensor.spi.GpuKernels;

import java.util.HashMap;
import java.util.Map;

/**
 * NVIDIA CUDA backend. All math runs in custom kernels that are compiled to PTX
 * at runtime with NVRTC and launched through the CUDA Driver API. Tensors stay
 * resident in device memory between operations.
 *
 * <p>Phase-0 note: a single driver context is created on first use; the notebook
 * exercises the backend from one thread. Per-thread primary-context handling is
 * tracked for a later phase.
 */
public final class CudaBackend implements ComputeBackend {

  private static final int BLOCK = 256;

  private boolean initialized;
  private CudaRuntime.DeviceInfo deviceInfo;
  private final Map<String, Long> functions = new HashMap<>();
  private final Map<String, Long> fusedFunctions = new HashMap<>();

  @Override
  public String name() {
    return "cuda";
  }

  @Override
  public DeviceType deviceType() {
    return DeviceType.CUDA;
  }

  @Override
  public boolean isAvailable() {
    try {
      return CudaRuntime.probe();
    } catch (Throwable failure) {
      return false;
    }
  }

  @Override
  public int priority() {
    return 100;
  }

  private synchronized void ensureInitialized() {
    if (initialized) {
      return;
    }
    deviceInfo = CudaRuntime.context();
    byte[] ptx = CudaRuntime.compile(GpuKernels.TENSOR_SOURCE, deviceInfo.arch());
    for (String kernel : GpuKernels.TENSOR_KERNEL_NAMES) {
      functions.put(kernel, CudaRuntime.loadFunction(ptx, kernel));
    }
    initialized = true;
  }

  public String deviceName() {
    ensureInitialized();
    return deviceInfo.name();
  }

  @Override
  public DeviceBuffer upload(float[] data, Shape shape, DType dtype, Device device) {
    ensureInitialized();
    requireCudaDevice(device);
    if (dtype != DType.F32) {
      throw new UnsupportedOperationException("CUDA float upload supports F32 only, got " + dtype);
    }
    // cuMemcpyHtoD is synchronous and is not ordered against the dedicated
    // non-blocking kernel stream. Retire prior uses before malloc can reuse a
    // pooled pointer for the next batch's upload.
    CudaRuntime.synchronize();
    return new CudaBuffer(CudaRuntime.uploadFloats(data), shape, dtype, device);
  }

  @Override
  public DeviceBuffer randomUniform(long seed, long counter, Shape shape, Device device) {
    ensureInitialized();
    requireCudaDevice(device);
    int size = Math.toIntExact(shape.size());
    CudaBuffer out = alloc(shape);
    CudaRuntime.launch(functions.get("random_uniform"), grid(size), BLOCK,
        out.pointer, seed, counter, size);
    return out;
  }

  @Override
  public DeviceBuffer randomNormal(long seed, long counter, Shape shape, Device device) {
    ensureInitialized();
    requireCudaDevice(device);
    int size = Math.toIntExact(shape.size());
    CudaBuffer out = alloc(shape);
    CudaRuntime.launch(functions.get("random_normal"), grid(size), BLOCK,
        out.pointer, seed, counter, size);
    return out;
  }

  @Override
  public float[] download(DeviceBuffer buffer) {
    ensureInitialized();
    CudaRuntime.synchronize();
    CudaBuffer cuda = cuda(buffer);
    return CudaRuntime.downloadFloats(cuda.pointer, cuda.count());
  }

  @Override
  public DeviceBuffer binary(Op op, DeviceBuffer a, DeviceBuffer b) {
    ensureInitialized();
    CudaBuffer left = cuda(a);
    CudaBuffer right = cuda(b);
    int n = left.count();
    if (n != right.count()) {
      throw new IllegalArgumentException("shape mismatch: " + left.shape() + " vs " + right.shape());
    }
    CudaBuffer out = alloc(left.shape());
    CudaRuntime.launch(functions.get("ew_binary"), grid(n), BLOCK,
        out.pointer, left.pointer, right.pointer, n, binaryCode(op));
    return out;
  }

  @Override
  public DeviceBuffer scalar(Op op, DeviceBuffer a, double value) {
    ensureInitialized();
    CudaBuffer left = cuda(a);
    int n = left.count();
    CudaBuffer out = alloc(left.shape());
    CudaRuntime.launch(functions.get("ew_scalar"), grid(n), BLOCK,
        out.pointer, left.pointer, (float) value, n, binaryCode(op));
    return out;
  }

  @Override
  public DeviceBuffer unary(Op op, DeviceBuffer a) {
    ensureInitialized();
    CudaBuffer left = cuda(a);
    int n = left.count();
    CudaBuffer out = alloc(left.shape());
    CudaRuntime.launch(functions.get("ew_unary"), grid(n), BLOCK,
        out.pointer, left.pointer, n, unaryCode(op));
    return out;
  }

  @Override
  public DeviceBuffer transpose(DeviceBuffer a) {
    ensureInitialized();
    CudaBuffer left = cuda(a);
    Shape shape = left.shape();
    if (shape.rank() != 2) {
      throw new IllegalStateException("transpose requires a rank-2 tensor, got " + shape);
    }
    int rows = shape.dim(0);
    int cols = shape.dim(1);
    CudaBuffer out = alloc(shape.transposed());
    CudaRuntime.launch(functions.get("transpose2d"), grid(rows * cols), BLOCK,
        out.pointer, left.pointer, rows, cols);
    return out;
  }

  @Override
  public DeviceBuffer matmul(DeviceBuffer a, DeviceBuffer b) {
    ensureInitialized();
    CudaBuffer left = cuda(a);
    CudaBuffer right = cuda(b);
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
    CudaBuffer out = alloc(Shape.of(m, n));
    long totalTiles = Math.multiplyExact((long) grid16(m), grid16(n));
    for (long offset = 0; offset < totalTiles; offset += Integer.MAX_VALUE) {
      int blocks = Math.toIntExact(Math.min(Integer.MAX_VALUE, totalTiles - offset));
      CudaRuntime.launchBlocks2d(functions.get("matmul_tiled"), blocks, 16, 16,
          out.pointer, left.pointer, right.pointer, m, k, n, offset);
    }
    return out;
  }

  @Override
  public DeviceBuffer batchedMatmul(DeviceBuffer a, DeviceBuffer b) {
    ensureInitialized();
    CudaBuffer left = cuda(a);
    CudaBuffer right = cuda(b);
    Shape ls = left.shape();
    Shape rs = right.shape();
    requireMatchingPlacement(left, right);
    if (ls.rank() != 3 || rs.rank() != 3
        || ls.dim(0) != rs.dim(0) || ls.dim(2) != rs.dim(1)) {
      throw new IllegalArgumentException("incompatible batched matmul shapes: " + ls + " x " + rs);
    }
    int batch = ls.dim(0);
    int m = ls.dim(1);
    int k = ls.dim(2);
    int n = rs.dim(2);
    requirePositiveMatmulDimensions(batch, m, k, n);
    CudaBuffer out = alloc(Shape.of(batch, m, n));
    long tilesPerBatch = Math.multiplyExact((long) grid16(m), grid16(n));
    long totalTiles = Math.multiplyExact((long) batch, tilesPerBatch);
    for (long offset = 0; offset < totalTiles; offset += Integer.MAX_VALUE) {
      int blocks = Math.toIntExact(Math.min(Integer.MAX_VALUE, totalTiles - offset));
      CudaRuntime.launchBlocks2d(functions.get("batched_matmul_tiled"), blocks, 16, 16,
          out.pointer, left.pointer, right.pointer, batch, m, k, n, offset);
    }
    return out;
  }

  @Override
  public DeviceBuffer sliceAxis0(DeviceBuffer input, int index) {
    ensureInitialized();
    CudaBuffer in = cuda(input);
    Shape shape = in.shape();
    if (shape.rank() < 1 || index < 0 || index >= shape.dim(0)) {
      throw new IllegalArgumentException("axis-0 index " + index + " is out of bounds for " + shape);
    }
    long sliceSize = shape.size() / shape.dim(0);
    int[] dims = shape.dims();
    int[] outputDims = new int[dims.length - 1];
    System.arraycopy(dims, 1, outputDims, 0, outputDims.length);
    CudaBuffer out = alloc(Shape.of(outputDims), in.dtype(), in.device());
    long bytes = Math.multiplyExact(sliceSize, in.dtype().byteSize());
    long sourceOffset = Math.multiplyExact(Math.multiplyExact((long) index, sliceSize),
        in.dtype().byteSize());
    CudaRuntime.copyDeviceAsync(out.pointer, Math.addExact(in.pointer, sourceOffset), bytes);
    return out;
  }

  @Override
  public DeviceBuffer stackAxis0(DeviceBuffer[] inputs) {
    ensureInitialized();
    if (inputs.length == 0) {
      throw new IllegalArgumentException("stackAxis0 requires at least one input");
    }
    CudaBuffer first = cuda(inputs[0]);
    long sliceSize = first.shape().size();
    int[] inputDims = first.shape().dims();
    int[] outputDims = new int[inputDims.length + 1];
    outputDims[0] = inputs.length;
    System.arraycopy(inputDims, 0, outputDims, 1, inputDims.length);
    CudaBuffer out = alloc(Shape.of(outputDims), first.dtype(), first.device());
    long bytes = Math.multiplyExact(sliceSize, first.dtype().byteSize());
    for (int i = 0; i < inputs.length; i++) {
      CudaBuffer input = cuda(inputs[i]);
      if (!input.shape().equals(first.shape())
          || input.dtype() != first.dtype()
          || !input.device().equals(first.device())) {
        release(out);
        throw new IllegalArgumentException("stackAxis0 inputs must have identical shape, dtype, and device");
      }
      long destinationOffset = Math.multiplyExact(Math.multiplyExact((long) i, sliceSize),
          first.dtype().byteSize());
      CudaRuntime.copyDeviceAsync(Math.addExact(out.pointer, destinationOffset), input.pointer, bytes);
    }
    return out;
  }

  private static int grid16(int size) {
    return 1 + (size - 1) / 16;
  }

  private static final int MAX_BROADCAST_RANK = 4;

  @Override
  public DeviceBuffer reduceSum(DeviceBuffer a) {
    ensureInitialized();
    CudaBuffer in = cuda(a);
    CudaBuffer out = alloc(Shape.of(1));
    CudaRuntime.launch(functions.get("reduce_sum"), 1, BLOCK, out.pointer, in.pointer, in.count());
    return out;
  }

  @Override
  public DeviceBuffer reduceMax(DeviceBuffer a) {
    ensureInitialized();
    CudaBuffer in = cuda(a);
    CudaBuffer out = alloc(Shape.of(1));
    CudaRuntime.launch(functions.get("reduce_max"), 1, BLOCK, out.pointer, in.pointer, in.count());
    return out;
  }

  @Override
  public DeviceBuffer conv2d(DeviceBuffer x, DeviceBuffer w, ConvSpec spec) {
    ensureInitialized();
    CudaBuffer in = cuda(x);
    CudaBuffer filters = cuda(w);
    int batch = in.shape().dim(0);
    CudaBuffer out = alloc(spec.outputShape(batch));
    int total = batch * spec.outputSize();
    CudaRuntime.launch(functions.get("conv2d_fwd"), grid(total), BLOCK,
        out.pointer, in.pointer, filters.pointer,
        batch, spec.inChannels(), spec.inHeight(), spec.inWidth(),
        spec.outChannels(), spec.kernel(), spec.stride(), spec.pad(),
        spec.outHeight(), spec.outWidth());
    return out;
  }

  @Override
  public DeviceBuffer conv2dGradInput(DeviceBuffer upstream, DeviceBuffer w, ConvSpec spec) {
    ensureInitialized();
    CudaBuffer grad = cuda(upstream);
    CudaBuffer filters = cuda(w);
    int batch = grad.shape().dim(0);
    CudaBuffer out = alloc(spec.inputShape(batch));
    int total = batch * spec.inputSize();
    CudaRuntime.launch(functions.get("conv2d_dx"), grid(total), BLOCK,
        out.pointer, grad.pointer, filters.pointer,
        batch, spec.inChannels(), spec.inHeight(), spec.inWidth(),
        spec.outChannels(), spec.kernel(), spec.stride(), spec.pad(),
        spec.outHeight(), spec.outWidth());
    return out;
  }

  @Override
  public DeviceBuffer conv2dGradWeight(DeviceBuffer x, DeviceBuffer upstream, ConvSpec spec) {
    ensureInitialized();
    CudaBuffer in = cuda(x);
    CudaBuffer grad = cuda(upstream);
    int batch = in.shape().dim(0);
    CudaBuffer out = alloc(spec.weightShape());
    int total = spec.outChannels() * spec.weightSize();
    CudaRuntime.launch(functions.get("conv2d_dw"), grid(total), BLOCK,
        out.pointer, in.pointer, grad.pointer,
        batch, spec.inChannels(), spec.inHeight(), spec.inWidth(),
        spec.outChannels(), spec.kernel(), spec.stride(), spec.pad(),
        spec.outHeight(), spec.outWidth());
    return out;
  }

  @Override
  public DeviceBuffer reluBackward(DeviceBuffer upstream, DeviceBuffer input) {
    ensureInitialized();
    CudaBuffer grad = cuda(upstream);
    CudaBuffer in = cuda(input);
    int n = grad.count();
    CudaBuffer out = alloc(grad.shape());
    CudaRuntime.launch(functions.get("relu_backward"), grid(n), BLOCK,
        out.pointer, grad.pointer, in.pointer, n);
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
    CudaBuffer grad = cuda(upstream);
    CudaBuffer in = cuda(input);
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
      throw new IllegalArgumentException(
          "maxAxisBackward gradient shape " + grad.shape()
              + " is incompatible with input " + shape + " and axis " + axis);
    }
    CudaBuffer out = alloc(shape);
    CudaRuntime.launch(
        functions.get("reduce_axis_max_backward"), grid(outer * inner), BLOCK,
        out.pointer, grad.pointer, in.pointer, outer, shape.dim(axis), inner);
    return out;
  }

  @Override
  public DeviceBuffer reshape(DeviceBuffer a, Shape target) {
    ensureInitialized();
    CudaBuffer input = cuda(a);
    CudaBuffer out = alloc(target);
    CudaRuntime.copyDeviceAsync(
        out.pointer, input.pointer, target.size() * input.dtype().byteSize());
    return out;
  }

  /**
   * Broadcasts entirely on-device: source strides (0 for dims being
   * broadcast) and target dims are computed on the host (cheap, just a few
   * ints) and passed as kernel scalars, capped at {@link #MAX_BROADCAST_RANK}
   * dims (padded with size-1/stride-0 entries) - every shape this project
   * produces is rank <= 2. No download/upload round trip, unlike the
   * previous host-side implementation.
   */
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

    // left-pad both dims and strides to MAX_BROADCAST_RANK with size-1/stride-0 entries
    int leadPad = MAX_BROADCAST_RANK - rank;
    int[] d = new int[MAX_BROADCAST_RANK];
    int[] s = new int[MAX_BROADCAST_RANK];
    for (int i = 0; i < MAX_BROADCAST_RANK; i++) {
      d[i] = i < leadPad ? 1 : targetDims[i - leadPad];
      s[i] = i < leadPad ? 0 : srcStrides[i - leadPad];
    }

    CudaBuffer in = cuda(a);
    CudaBuffer out = alloc(target);
    int n = (int) target.size();
    CudaRuntime.launch(functions.get("broadcast_to"), grid(n), BLOCK,
        out.pointer, in.pointer, d[0], d[1], d[2], d[3], s[0], s[1], s[2], s[3], n);
    return out;
  }

  @Override
  public void sync() {
    if (initialized) {
      CudaRuntime.synchronize();
    }
  }

  @Override
  public void release(DeviceBuffer buffer) {
    cuda(buffer).release();
  }

  /**
   * Runs a whole elementwise expression chain as one generated kernel launch,
   * instead of one launch (and one intermediate device buffer) per op. The
   * compiled kernel is cached by its generated source, so repeated fused
   * calls for the same op chain (any shape) reuse it without recompiling.
   */
  @Override
  public DeviceBuffer fused(Expr expr, DeviceBuffer[] inputs) {
    ensureInitialized();
    CudaBuffer[] buffers = new CudaBuffer[inputs.length];
    int n = -1;
    for (int i = 0; i < inputs.length; i++) {
      buffers[i] = cuda(inputs[i]);
      if (i == 0) {
        n = buffers[i].count();
      } else if (buffers[i].count() != n) {
        throw new IllegalArgumentException("fused inputs must share element count");
      }
    }
    String source = GpuKernels.fusedSource(expr, inputs.length);
    long function = fusedFunctions.computeIfAbsent(source,
        src -> CudaRuntime.loadFunction(
            CudaRuntime.compile(src, deviceInfo.arch()), GpuKernels.FUSED_KERNEL_NAME));
    CudaBuffer out = alloc(buffers[0].shape());
    Object[] arguments = new Object[inputs.length + 2];
    arguments[0] = out.pointer;
    for (int i = 0; i < inputs.length; i++) {
      arguments[i + 1] = buffers[i].pointer;
    }
    arguments[inputs.length + 1] = n;
    CudaRuntime.launch(function, grid(n), BLOCK, arguments);
    return out;
  }

  private CudaBuffer alloc(Shape shape) {
    return alloc(shape, DType.F32, Device.cuda());
  }

  private CudaBuffer alloc(Shape shape, DType dtype, Device device) {
    requireCudaDevice(device);
    long bytes = Math.multiplyExact(shape.size(), dtype.byteSize());
    return new CudaBuffer(CudaRuntime.malloc(bytes), shape, dtype, device);
  }

  private static int grid(int n) {
    return n == 0 ? 0 : 1 + (n - 1) / BLOCK;
  }

  private static void requirePositiveMatmulDimensions(int... dimensions) {
    for (int dimension : dimensions) {
      if (dimension < 1) {
        throw new IllegalArgumentException("matmul dimensions must be positive");
      }
    }
  }

  private static void requireCudaDevice(Device device) {
    if (!device.equals(Device.cuda())) {
      throw new IllegalArgumentException("CUDA backend currently supports only cuda:0, got " + device);
    }
  }

  private static void requireMatchingPlacement(CudaBuffer left, CudaBuffer right) {
    if (left.dtype() != right.dtype() || !left.device().equals(right.device())) {
      throw new IllegalArgumentException("matmul operands must use the same device and dtype");
    }
    requireCudaDevice(left.device());
    if (left.dtype() != DType.F32) {
      throw new UnsupportedOperationException("CUDA matmul supports F32 only, got " + left.dtype());
    }
  }

  private static CudaBuffer cuda(DeviceBuffer buffer) {
    if (buffer instanceof CudaBuffer cuda) {
      return cuda;
    }
    throw new IllegalArgumentException("expected a CUDA buffer, got " + buffer.getClass());
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

  private DeviceBuffer reduceAxis(
      DeviceBuffer buffer, int axis, boolean keepDims, String kernel) {
    ensureInitialized();
    CudaBuffer input = cuda(buffer);
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
    CudaBuffer out = alloc(Shape.of(outputDims));
    int outputSize = outer * inner;
    CudaRuntime.launch(
        functions.get(kernel), grid(outputSize), BLOCK,
        out.pointer, input.pointer, outer, shape.dim(axis), inner);
    return out;
  }
}
