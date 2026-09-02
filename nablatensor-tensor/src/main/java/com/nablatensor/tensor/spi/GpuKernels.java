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

import com.nablatensor.tensor.Op;
import com.nablatensor.tensor.expr.Expr;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The compute-kernel source shared by the CUDA and ROCm/HIP tensor backends.
 *
 * <p>The two backends run byte-for-byte the same kernels: the bodies are plain
 * CUDA C that NVRTC compiles to PTX and HIPRTC accepts unchanged and compiles to
 * a GCN code object. Keeping the source here rather than copied into each backend
 * means an op added or a bug fixed lands on both GPU flavours at once. (The
 * Vulkan backend is unrelated — it runs GLSL compute shaders through shaderc.)
 *
 * <p>This class is a façade. The kernels themselves live one family per file
 * ({@link ElementwiseKernels}, {@link MatmulKernels}, {@link RandomKernels},
 * {@link ReductionKernels}, {@link ConvKernels}), each as a {@link GpuKernel}
 * that parses its own entry-point name and parameter types out of its source —
 * so {@link #TENSOR_KERNEL_NAMES} is derived, never hand-maintained, and cannot
 * drift from the code it names.
 *
 * <p>If a kernel ever needs architecture-specific tuning that CUDA C's shared
 * subset cannot express (wavefront-64 vs warp-32 reductions, differing block
 * sizes, {@code __syncwarp}), split that one kernel back out into the backend
 * that needs the variant; everything that stays identical stays here.
 */
public final class GpuKernels {

  private GpuKernels() {
  }

  /** Entry-point name of the kernel {@link #fusedSource} generates. */
  public static final String FUSED_KERNEL_NAME = "fused_kernel";

  /** Every shared kernel, by family, in the order they are emitted and loaded. */
  public static final List<GpuKernel> TENSOR_KERNELS = Stream.of(
          ElementwiseKernels.KERNELS, MatmulKernels.KERNELS, RandomKernels.KERNELS,
          ReductionKernels.KERNELS, ConvKernels.KERNELS)
      .flatMap(List::stream)
      .toList();

  private static final Map<String, GpuKernel> BY_NAME = indexByName(TENSOR_KERNELS);

  /** Kernels in {@link #TENSOR_SOURCE}, in a stable order for module loading. */
  public static final String[] TENSOR_KERNEL_NAMES = BY_NAME.keySet().toArray(String[]::new);

  /** Every static tensor kernel, as one CUDA-C translation unit. */
  public static final String TENSOR_SOURCE = DevicePrelude.SOURCE
      + TENSOR_KERNELS.stream().map(GpuKernel::source).collect(Collectors.joining());

  /** Looks up a shared kernel by the entry-point name parsed out of its source. */
  public static GpuKernel kernel(String name) {
    GpuKernel kernel = BY_NAME.get(name);
    if (kernel == null) {
      throw new IllegalArgumentException("no shared GPU kernel named " + name);
    }
    return kernel;
  }

  private static Map<String, GpuKernel> indexByName(List<GpuKernel> kernels) {
    Map<String, GpuKernel> byName = new LinkedHashMap<>();
    for (GpuKernel kernel : kernels) {
      if (byName.put(kernel.name(), kernel) != null) {
        throw new IllegalStateException("duplicate GPU kernel name: " + kernel.name());
      }
    }
    return Collections.unmodifiableMap(byName);
  }

  /**
   * Emits a whole elementwise expression chain as one {@code fused_kernel}
   * translation unit, so a backend can run the chain in a single launch with no
   * per-op intermediate buffers. The text is CUDA C; HIPRTC takes it unchanged.
   */
  public static String fusedSource(Expr expr, int numInputs) {
    StringBuilder body = new StringBuilder();
    Map<Expr, String> names = new IdentityHashMap<>();
    String result = emitFused(expr, names, body);
    StringBuilder source = new StringBuilder(
        "extern \"C\" __global__ void " + FUSED_KERNEL_NAME + "(float* out");
    for (int i = 0; i < numInputs; i++) {
      source.append(", const float* in").append(i);
    }
    source.append(", int n) {\n")
        .append("  int i = blockIdx.x * blockDim.x + threadIdx.x;\n")
        .append("  if (i >= n) return;\n")
        .append(body)
        .append("  out[i] = ").append(result).append(";\n")
        .append("}\n");
    return source.toString();
  }

  /**
   * Emits one SSA-style local per unique node (shared subtrees reuse their
   * name), post-order, so a diamond in the expression tree is computed once. The
   * name is allocated from {@code names.size()} only after recursing into
   * children, so each node's number reflects how many distinct nodes were
   * already emitted.
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
        body.append("  float ").append(name).append(" = in")
            .append(in.index()).append("[i];\n");
      }
      case Expr.Unary u -> {
        String x = emitFused(u.in(), names, body);
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = ").append(unaryExpr(u.op(), x)).append(";\n");
      }
      case Expr.Binary b -> {
        String l = emitFused(b.left(), names, body);
        String r = emitFused(b.right(), names, body);
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = ").append(binaryExpr(b.op(), l, r)).append(";\n");
      }
      case Expr.Scalar s -> {
        String x = emitFused(s.in(), names, body);
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = ")
            .append(binaryExpr(s.op(), x, literal(s.value()))).append(";\n");
      }
    }
    names.put(expr, name);
    return name;
  }

  private static String literal(double value) {
    return Float.toString((float) value) + "f";
  }

  private static String binaryExpr(Op op, String l, String r) {
    return switch (op) {
      case ADD -> "(" + l + " + " + r + ")";
      case SUB -> "(" + l + " - " + r + ")";
      case MUL -> "(" + l + " * " + r + ")";
      case DIV -> "(" + l + " / " + r + ")";
      case MAX -> "(isnan(" + l + ") ? (" + l + ") : (isnan(" + r + ") ? (" + r
          + ") : fmaxf(" + l + ", " + r + ")))";
      case MIN -> "(isnan(" + l + ") ? (" + l + ") : (isnan(" + r + ") ? (" + r
          + ") : fminf(" + l + ", " + r + ")))";
      default -> throw new IllegalArgumentException("not a binary/scalar op: " + op);
    };
  }

  private static String unaryExpr(Op op, String x) {
    return switch (op) {
      case NEG -> "(-" + x + ")";
      case EXP -> "expf(" + x + ")";
      case LOG -> "logf(" + x + ")";
      case SQRT -> "sqrtf(" + x + ")";
      case RSQRT -> "rsqrtf(" + x + ")";
      case TANH -> "tanhf(" + x + ")";
      case SIGMOID -> "(1.0f / (1.0f + expf(-" + x + ")))";
      case RELU -> "(isnan(" + x + ") ? (" + x + ") : fmaxf(0.0f, " + x + "))";
      case ABS -> "fabsf(" + x + ")";
      case SIGN -> "((" + x + ") > 0.0f ? 1.0f : ((" + x + ") < 0.0f ? -1.0f : 0.0f))";
      default -> throw new IllegalArgumentException("not a unary op: " + op);
    };
  }
}
