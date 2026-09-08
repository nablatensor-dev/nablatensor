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
package com.nablatensor.backend.vulkan;

import com.nablatensor.tensor.ConvSpec;
import com.nablatensor.tensor.DType;
import com.nablatensor.tensor.Device;
import com.nablatensor.tensor.DeviceType;
import com.nablatensor.tensor.Op;
import com.nablatensor.tensor.Shape;
import com.nablatensor.tensor.expr.Expr;
import com.nablatensor.tensor.spi.AxisReduction;
import com.nablatensor.tensor.spi.BroadcastLayout;
import com.nablatensor.tensor.spi.ComputeBackend;
import com.nablatensor.tensor.spi.DeviceBuffer;
import com.nablatensor.tensor.spi.GpuLaunch;
import com.nablatensor.tensor.spi.OpCodes;

import java.util.IdentityHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Vulkan compute backend. Math runs in GLSL compute shaders compiled to SPIR-V
 * at runtime with {@code libshaderc} and dispatched through the Vulkan loader
 * via {@link java.lang.foreign}; tensors stay resident in device memory
 * (host-visible on an APU) between operations.
 *
 * <p>Elementwise (including generated fused chains) / reductions / matmul /
 * transpose / broadcast / relu-backward / 2-D convolution are implemented. The
 * shader library is {@link VulkanShaders}; this class is op dispatch only.
 */
public final class VulkanBackend implements ComputeBackend {

  private static final int BLOCK = GpuLaunch.DEFAULT_BLOCK;

  /** Phase-1 workgroup count for two-phase reductions; also the partials-buffer length. */
  private static final int REDUCE_GROUPS = 256;

  private volatile boolean initialized;

  /**
   * Generated fused-elementwise shader source -&gt; the registered {@link
   * VulkanShader}. Keyed by source text so two op chains that emit the same GLSL
   * share one compiled pipeline, and a shape change (element count is a push
   * constant, not baked into the source) never recompiles.
   */
  private final ConcurrentHashMap<String, VulkanShader> fusedPipelines = new ConcurrentHashMap<>();

  private static final AtomicInteger FUSED_SEQ = new AtomicInteger();

  @Override
  public String name() {
    return "vulkan";
  }

  @Override
  public DeviceType deviceType() {
    return DeviceType.VULKAN;
  }

  @Override
  public boolean isAvailable() {
    try {
      return VulkanRuntime.probe();
    } catch (Throwable failure) {
      return false;
    }
  }

  @Override
  public int priority() {
    // Above ROCm (50): on the machines this project targets Vulkan (Mesa RADV)
    // is the faster compute path, and this matches the AAD-engine ordering
    // (vulkan 60 > rocm 55). Still below CUDA (100).
    return 60;
  }

  public String deviceName() {
    return VulkanRuntime.deviceName();
  }

  private synchronized void ensureInitialized() {
    if (initialized) {
      return;
    }
    VulkanRuntime.init();
    for (VulkanShader shader : VulkanShaders.ALL) {
      register(shader);
    }
    initialized = true;
  }

  private static void register(VulkanShader shader) {
    VulkanRuntime.registerPipeline(shader.name(), shader.source(), shader.bindings());
  }

  /** Records one dispatch of {@code shader}, keyed by its own {@link VulkanShader#name()}. */
  private static void dispatch(VulkanShader shader, int groupsX, int groupsY, int groupsZ,
      long[] buffers, int[] pushInts, int pushFloatSlot, float pushFloat) {
    VulkanRuntime.dispatch(shader.name(), groupsX, groupsY, groupsZ,
        buffers, pushInts, pushFloatSlot, pushFloat);
  }

  // ---- data movement ---------------------------------------------------------

  @Override
  public DeviceBuffer upload(float[] data, Shape shape, DType dtype, Device device) {
    ensureInitialized();
    requireVulkanDevice(device);
    if (dtype != DType.F32) {
      throw new UnsupportedOperationException("Vulkan float upload supports F32 only, got " + dtype);
    }
    long[] handles = VulkanRuntime.alloc(Math.max(4L, (long) data.length * Float.BYTES));
    VulkanRuntime.writeFloats(handles[1], data);
    return new VulkanBuffer(handles[0], handles[1], shape, dtype, device);
  }

  @Override
  public float[] download(DeviceBuffer buffer) {
    ensureInitialized();
    VulkanRuntime.sync();
    VulkanBuffer vk = vk(buffer);
    return VulkanRuntime.readFloats(vk.memory, vk.count());
  }

  // ---- elementwise ---------------------------------------------------------

  @Override
  public DeviceBuffer binary(Op op, DeviceBuffer a, DeviceBuffer b) {
    ensureInitialized();
    VulkanBuffer left = vk(a);
    VulkanBuffer right = vk(b);
    int n = left.count();
    if (n != right.count()) {
      throw new IllegalArgumentException("shape mismatch: " + left.shape() + " vs " + right.shape());
    }
    VulkanBuffer out = alloc(left.shape());
    dispatch(VulkanShaders.EW_BINARY, grid(n), 1, 1,
        new long[] {out.buffer, left.buffer, right.buffer}, new int[] {n, OpCodes.binary(op)}, -1, 0f);
    return out;
  }

  @Override
  public DeviceBuffer scalar(Op op, DeviceBuffer a, double value) {
    ensureInitialized();
    VulkanBuffer left = vk(a);
    int n = left.count();
    VulkanBuffer out = alloc(left.shape());
    dispatch(VulkanShaders.EW_SCALAR, grid(n), 1, 1,
        new long[] {out.buffer, left.buffer}, new int[] {n, OpCodes.binary(op)}, 2, (float) value);
    return out;
  }

  @Override
  public DeviceBuffer unary(Op op, DeviceBuffer a) {
    ensureInitialized();
    VulkanBuffer left = vk(a);
    int n = left.count();
    VulkanBuffer out = alloc(left.shape());
    dispatch(VulkanShaders.EW_UNARY, grid(n), 1, 1,
        new long[] {out.buffer, left.buffer}, new int[] {n, OpCodes.unary(op)}, -1, 0f);
    return out;
  }

  // ---- fused elementwise -------------------------------------------------

  /**
   * Runs a whole elementwise {@link Expr} chain as one generated GLSL compute
   * shader (one dispatch, one intermediate buffer), instead of the SPI default
   * of one dispatch and one buffer per node. The shader is emitted with the
   * same SSA post-order walk as the CUDA backend, compiled to SPIR-V once by
   * {@code libshaderc}, and cached (as a registered pipeline) by its source.
   */
  @Override
  public DeviceBuffer fused(Expr expr, DeviceBuffer[] inputs) {
    ensureInitialized();
    if (inputs.length == 0) {
      throw new IllegalArgumentException("fused requires at least one input");
    }
    VulkanBuffer[] operands = new VulkanBuffer[inputs.length];
    int n = -1;
    for (int i = 0; i < inputs.length; i++) {
      operands[i] = vk(inputs[i]);
      if (i == 0) {
        n = operands[i].count();
      } else if (operands[i].count() != n) {
        throw new IllegalArgumentException("fused inputs must share element count");
      }
    }
    int numInputs = inputs.length;
    String source = generateFusedGlsl(expr, numInputs);
    VulkanShader pipeline = fusedPipelines.computeIfAbsent(source, src -> {
      VulkanShader shader = VulkanShader.of("fused_" + FUSED_SEQ.getAndIncrement(), src);
      register(shader);
      return shader;
    });
    VulkanBuffer out = alloc(operands[0].shape());
    long[] handles = new long[numInputs + 1];
    handles[0] = out.buffer;
    for (int i = 0; i < numInputs; i++) {
      handles[i + 1] = operands[i].buffer;
    }
    dispatch(pipeline, grid(n), 1, 1, handles, new int[] {n}, -1, 0f);
    return out;
  }

  private static String generateFusedGlsl(Expr expr, int numInputs) {
    StringBuilder body = new StringBuilder();
    String result = emitFused(expr, new IdentityHashMap<>(), body);
    StringBuilder src = new StringBuilder(256);
    src.append(VulkanShaders.COMMON)
        .append("layout(local_size_x = 256) in;\n")
        .append("layout(std430, binding = 0) writeonly buffer O { float o[]; };\n");
    for (int i = 0; i < numInputs; i++) {
      src.append("layout(std430, binding = ").append(i + 1)
          .append(") readonly buffer I").append(i)
          .append(" { float in").append(i).append("[]; };\n");
    }
    src.append("layout(push_constant) uniform P { uint n; } p;\n")
        .append("void main() {\n")
        .append("  uint i = gl_GlobalInvocationID.x;\n")
        .append("  if (i >= p.n) return;\n")
        .append(body)
        .append("  o[i] = ").append(result).append(";\n")
        .append("}\n");
    return src.toString();
  }

  /**
   * Emits one SSA-style local per unique node (shared subtrees reuse their
   * name), post-order. The name is taken from {@code names.size()} only after
   * the children have been emitted, so each node's number reflects how many
   * distinct nodes were already written.
   */
  private static String emitFused(Expr expr, Map<Expr, String> names, StringBuilder body) {
    String cached = names.get(expr);
    if (cached != null) {
      return cached;
    }
    String name;
    switch (expr) {
      case Expr.Input in -> {
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = in").append(in.index()).append("[i];\n");
      }
      case Expr.Unary u -> {
        String x = emitFused(u.in(), names, body);
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = ").append(glslUnary(u.op(), x)).append(";\n");
      }
      case Expr.Binary b -> {
        String l = emitFused(b.left(), names, body);
        String r = emitFused(b.right(), names, body);
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = ").append(glslBinary(b.op(), l, r)).append(";\n");
      }
      case Expr.Scalar s -> {
        String x = emitFused(s.in(), names, body);
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = ")
            .append(glslBinary(s.op(), x, glslLiteral(s.value()))).append(";\n");
      }
    }
    names.put(expr, name);
    return name;
  }

  private static String glslLiteral(double value) {
    float f = (float) value;
    if (Float.isNaN(f)) {
      return "uintBitsToFloat(0x7fc00000u)";
    }
    if (f == Float.POSITIVE_INFINITY) {
      return "uintBitsToFloat(0x7f800000u)";
    }
    if (f == Float.NEGATIVE_INFINITY) {
      return "uintBitsToFloat(0xff800000u)";
    }
    return Float.toString(f);   // always contains '.' or an exponent -> a valid GLSL float
  }

  private static String glslBinary(Op op, String l, String r) {
    return switch (op) {
      case ADD -> "(" + l + " + " + r + ")";
      case SUB -> "(" + l + " - " + r + ")";
      case MUL -> "(" + l + " * " + r + ")";
      case DIV -> "(" + l + " / " + r + ")";
      case MAX -> "(isnan(" + l + ") ? (" + l + ") : (isnan(" + r + ") ? (" + r
          + ") : max(" + l + ", " + r + ")))";
      case MIN -> "(isnan(" + l + ") ? (" + l + ") : (isnan(" + r + ") ? (" + r
          + ") : min(" + l + ", " + r + ")))";
      default -> throw new IllegalArgumentException("not a binary/scalar op: " + op);
    };
  }

  private static String glslUnary(Op op, String x) {
    return switch (op) {
      case NEG -> "(-" + x + ")";
      case EXP -> "exp(" + x + ")";
      case LOG -> "log(" + x + ")";
      case SQRT -> "sqrt(" + x + ")";
      case RSQRT -> "inversesqrt(" + x + ")";
      case TANH -> "tanh(" + x + ")";
      case SIGMOID -> "(1.0 / (1.0 + exp(-(" + x + "))))";
      case RELU -> "(isnan(" + x + ") ? (" + x + ") : max(0.0, " + x + "))";
      case ABS -> "abs(" + x + ")";
      case SIGN -> "((" + x + ") > 0.0 ? 1.0 : ((" + x + ") < 0.0 ? -1.0 : 0.0))";
      default -> throw new IllegalArgumentException("not a unary op: " + op);
    };
  }

  // ---- linear algebra ----------------------------------------------------

  @Override
  public DeviceBuffer transpose(DeviceBuffer a) {
    ensureInitialized();
    VulkanBuffer left = vk(a);
    Shape shape = left.shape();
    if (shape.rank() != 2) {
      throw new IllegalStateException("transpose requires a rank-2 tensor, got " + shape);
    }
    int rows = shape.dim(0);
    int cols = shape.dim(1);
    VulkanBuffer out = alloc(shape.transposed());
    dispatch(VulkanShaders.TRANSPOSE, grid(rows * cols), 1, 1,
        new long[] {out.buffer, left.buffer}, new int[] {rows, cols}, -1, 0f);
    return out;
  }

  @Override
  public DeviceBuffer matmul(DeviceBuffer a, DeviceBuffer b) {
    ensureInitialized();
    VulkanBuffer left = vk(a);
    VulkanBuffer right = vk(b);
    Shape ls = left.shape();
    Shape rs = right.shape();
    requireMatchingPlacement(left, right);
    if (ls.rank() != 2 || rs.rank() != 2 || ls.dim(1) != rs.dim(0)) {
      throw new IllegalArgumentException("incompatible matmul shapes: " + ls + " x " + rs);
    }
    int m = ls.dim(0);
    int k = ls.dim(1);
    int n = rs.dim(1);
    if (m < 1 || k < 1 || n < 1) {
      throw new IllegalArgumentException("matmul dimensions must be positive");
    }
    VulkanBuffer out = alloc(Shape.of(m, n));
    dispatch(VulkanShaders.MATMUL, GpuLaunch.ceilDiv(n, VulkanShaders.MM_BN), GpuLaunch.ceilDiv(m, VulkanShaders.MM_BM), 1,
        new long[] {out.buffer, left.buffer, right.buffer}, new int[] {m, k, n}, -1, 0f);
    return out;
  }

  // ---- reductions ------------------------------------------------------------

  @Override
  public DeviceBuffer reduceSum(DeviceBuffer a) {
    return reduce(vk(a), VulkanShaders.REDUCE_SUM_P1, VulkanShaders.REDUCE_SUM);
  }

  @Override
  public DeviceBuffer reduceMax(DeviceBuffer a) {
    return reduce(vk(a), VulkanShaders.REDUCE_MAX_P1, VulkanShaders.REDUCE_MAX);
  }

  /**
   * Two-phase whole-tensor reduction: phase 1 spreads the input over
   * {@value #REDUCE_GROUPS} workgroups, each emitting one partial; phase 2 folds
   * the partials in a single workgroup. Small inputs (one phase-1 workgroup or
   * fewer) skip straight to the single-workgroup path.
   */
  private DeviceBuffer reduce(VulkanBuffer in, VulkanShader phase1, VulkanShader phase2) {
    ensureInitialized();
    int n = in.count();
    int groups = Math.min(REDUCE_GROUPS, Math.max(1, grid(n)));
    VulkanBuffer out = alloc(Shape.of(1));
    if (groups <= 1) {
      dispatch(phase2, 1, 1, 1, new long[] {out.buffer, in.buffer}, new int[] {n}, -1, 0f);
      return out;
    }
    VulkanBuffer partials = alloc(Shape.of(groups));
    dispatch(phase1, groups, 1, 1,
        new long[] {partials.buffer, in.buffer}, new int[] {n}, -1, 0f);
    dispatch(phase2, 1, 1, 1,
        new long[] {out.buffer, partials.buffer}, new int[] {groups}, -1, 0f);
    partials.release();   // dispatch() blocks on vkQueueWaitIdle, so phase 2 is done reading
    return out;
  }

  @Override
  public DeviceBuffer sumAxis0(DeviceBuffer a) {
    ensureInitialized();
    VulkanBuffer in = vk(a);
    Shape shape = in.shape();
    if (shape.rank() != 2) {
      throw new IllegalStateException("sumAxis0 requires a rank-2 tensor, got " + shape);
    }
    int rows = shape.dim(0);
    int cols = shape.dim(1);
    VulkanBuffer out = alloc(Shape.of(cols));
    dispatch(VulkanShaders.SUM_AXIS0, grid(cols), 1, 1,
        new long[] {out.buffer, in.buffer}, new int[] {rows, cols}, -1, 0f);
    return out;
  }

  @Override
  public DeviceBuffer sumAxis(DeviceBuffer a, int axis, boolean keepDims) {
    return reduceAxis(a, axis, keepDims, VulkanShaders.REDUCE_AXIS_SUM);
  }

  @Override
  public DeviceBuffer maxAxis(DeviceBuffer a, int axis, boolean keepDims) {
    return reduceAxis(a, axis, keepDims, VulkanShaders.REDUCE_AXIS_MAX);
  }

  @Override
  public DeviceBuffer argMaxAxis(DeviceBuffer a, int axis) {
    return reduceAxis(a, axis, false, VulkanShaders.REDUCE_AXIS_ARGMAX);
  }

  @Override
  public DeviceBuffer maxAxisBackward(DeviceBuffer upstream, DeviceBuffer input, int axis) {
    ensureInitialized();
    VulkanBuffer grad = vk(upstream);
    VulkanBuffer in = vk(input);
    AxisReduction r = AxisReduction.of(in.shape(), axis);
    r.requireGradient(grad.shape(), grad.count());
    VulkanBuffer out = alloc(in.shape());
    dispatch(VulkanShaders.REDUCE_AXIS_MAX_BACKWARD, grid(r.outputSize()), 1, 1,
        new long[] {out.buffer, grad.buffer, in.buffer},
        new int[] {r.outer(), r.axisSize(), r.inner()}, -1, 0f);
    return out;
  }

  private DeviceBuffer reduceAxis(DeviceBuffer buffer, int axis, boolean keepDims, VulkanShader pipeline) {
    ensureInitialized();
    VulkanBuffer input = vk(buffer);
    AxisReduction r = AxisReduction.of(input.shape(), axis);
    VulkanBuffer out = alloc(r.outputShape(keepDims));
    dispatch(pipeline, grid(r.outputSize()), 1, 1,
        new long[] {out.buffer, input.buffer},
        new int[] {r.outer(), r.axisSize(), r.inner()}, -1, 0f);
    return out;
  }

  // ---- other ops -------------------------------------------------------------

  @Override
  public DeviceBuffer reluBackward(DeviceBuffer upstream, DeviceBuffer input) {
    ensureInitialized();
    VulkanBuffer grad = vk(upstream);
    VulkanBuffer in = vk(input);
    int n = grad.count();
    VulkanBuffer out = alloc(grad.shape());
    dispatch(VulkanShaders.RELU_BACKWARD, grid(n), 1, 1,
        new long[] {out.buffer, grad.buffer, in.buffer}, new int[] {n}, -1, 0f);
    return out;
  }

  @Override
  public DeviceBuffer broadcastTo(DeviceBuffer a, Shape target) {
    ensureInitialized();
    Shape src = a.shape();
    if (src.equals(target)) {
      return a;
    }
    BroadcastLayout layout = BroadcastLayout.of(src, target);
    int[] d = layout.dims();
    int[] s = layout.strides();
    VulkanBuffer in = vk(a);
    int n = Math.toIntExact(target.size());
    VulkanBuffer out = alloc(target);
    dispatch(VulkanShaders.BROADCAST, grid(n), 1, 1, new long[] {out.buffer, in.buffer},
        new int[] {d[0], d[1], d[2], d[3], s[0], s[1], s[2], s[3], n}, -1, 0f);
    return out;
  }

  private static int[] convPush(int batch, ConvSpec spec) {
    return new int[] {batch, spec.inChannels(), spec.inHeight(), spec.inWidth(),
        spec.outChannels(), spec.kernel(), spec.stride(), spec.pad(),
        spec.outHeight(), spec.outWidth()};
  }

  @Override
  public DeviceBuffer conv2d(DeviceBuffer x, DeviceBuffer w, ConvSpec spec) {
    ensureInitialized();
    VulkanBuffer in = vk(x);
    VulkanBuffer filters = vk(w);
    int batch = in.shape().dim(0);
    VulkanBuffer out = alloc(spec.outputShape(batch));
    dispatch(VulkanShaders.CONV2D_FWD, grid(batch * spec.outputSize()), 1, 1,
        new long[] {out.buffer, in.buffer, filters.buffer}, convPush(batch, spec), -1, 0f);
    return out;
  }

  @Override
  public DeviceBuffer conv2dGradInput(DeviceBuffer upstream, DeviceBuffer w, ConvSpec spec) {
    ensureInitialized();
    VulkanBuffer grad = vk(upstream);
    VulkanBuffer filters = vk(w);
    int batch = grad.shape().dim(0);
    VulkanBuffer out = alloc(spec.inputShape(batch));
    dispatch(VulkanShaders.CONV2D_DX, grid(batch * spec.inputSize()), 1, 1,
        new long[] {out.buffer, grad.buffer, filters.buffer}, convPush(batch, spec), -1, 0f);
    return out;
  }

  @Override
  public DeviceBuffer conv2dGradWeight(DeviceBuffer x, DeviceBuffer upstream, ConvSpec spec) {
    ensureInitialized();
    VulkanBuffer in = vk(x);
    VulkanBuffer grad = vk(upstream);
    int batch = in.shape().dim(0);
    VulkanBuffer out = alloc(spec.weightShape());
    dispatch(VulkanShaders.CONV2D_DW, grid(spec.outChannels() * spec.weightSize()), 1, 1,
        new long[] {out.buffer, in.buffer, grad.buffer}, convPush(batch, spec), -1, 0f);
    return out;
  }

  @Override
  public void sync() {
    if (initialized) {
      VulkanRuntime.sync();
    }
  }

  @Override
  public void release(DeviceBuffer buffer) {
    vk(buffer).release();
  }

  // ---- helpers -----------------------------------------------------------

  private VulkanBuffer alloc(Shape shape) {
    long bytes = Math.max(4L, Math.multiplyExact(shape.size(), (long) DType.F32.byteSize()));
    long[] handles = VulkanRuntime.alloc(bytes);
    return new VulkanBuffer(handles[0], handles[1], shape, DType.F32, Device.vulkan());
  }

  private static int grid(int n) {
    return GpuLaunch.grid1d(n);
  }

  private static void requireVulkanDevice(Device device) {
    if (!device.equals(Device.vulkan())) {
      throw new IllegalArgumentException("Vulkan backend currently supports only vulkan:0, got " + device);
    }
  }

  private static void requireMatchingPlacement(VulkanBuffer left, VulkanBuffer right) {
    if (left.dtype() != right.dtype() || !left.device().equals(right.device())) {
      throw new IllegalArgumentException("matmul operands must use the same device and dtype");
    }
    requireVulkanDevice(left.device());
    if (left.dtype() != DType.F32) {
      throw new UnsupportedOperationException("Vulkan matmul supports F32 only, got " + left.dtype());
    }
  }

  private static VulkanBuffer vk(DeviceBuffer buffer) {
    if (buffer instanceof VulkanBuffer vulkan) {
      return vulkan;
    }
    throw new IllegalArgumentException("expected a Vulkan buffer, got " + buffer.getClass());
  }


}
