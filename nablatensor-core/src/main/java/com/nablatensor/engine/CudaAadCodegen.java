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
 * Translates a recorded {@link AadTape} into CUDA C source implementing one
 * scenario per thread: a fully unrolled forward sweep followed by the adjoint
 * sweep, both in registers, with the per-scenario random stream generated
 * in-thread so a replay touches no memory beyond the handful of inputs and the
 * per-block reduction.
 *
 * <p>The emitted source is portable CUDA C: {@code AadKernel} compiles it with
 * NVRTC, and the ROCm engine ({@code com.nablatensor.engine.rocm}) compiles the exact
 * same string with HIPRTC, which accepts it unchanged. Kept {@code public} for
 * that second consumer; not part of the supported API.
 */
@Internal
public final class CudaAadCodegen {

  public static final String KERNEL_NAME = "aad_replay";
  public static final int BLOCK = 256;

  private CudaAadCodegen() {
  }

  public static String generate(AadTape tape, AadOptions options) {
    boolean f32 = options.precision() == AadOptions.Precision.FLOAT32;
    String real = f32 ? "float" : "double";
    StringBuilder src = new StringBuilder(1 << 16);

    src.append("#define BLOCK ").append(BLOCK).append('\n');
    src.append("typedef ").append(real).append(" real;\n");
    appendRng(src, f32);

    src.append("extern \"C\" __global__ void ").append(KERNEL_NAME).append("(\n")
        .append("    const double* __restrict__ inputs,\n")
        .append("    unsigned long long nPaths,\n")
        .append("    unsigned long long pathOffset,\n")
        .append("    unsigned long long seed,\n")
        .append("    double* __restrict__ partials) {\n");

    int nIn = tape.inputCount();
    for (int j = 0; j < nIn; j++) {
      src.append("  const real in").append(j).append(" = (real) inputs[").append(j).append("];\n");
    }
    src.append("  double accValue = 0.0;\n");
    if (options.adjoints()) {
      for (int j = 0; j < nIn; j++) {
        src.append("  double accAdj").append(j).append(" = 0.0;\n");
      }
    }
    src.append("  const unsigned long long stride = (unsigned long long) blockDim.x * gridDim.x;\n")
        .append("  for (unsigned long long path = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;\n")
        .append("       path < nPaths; path += stride) {\n")
        .append("    Rng rng; rng_init(rng, path + pathOffset, seed);\n");

    emitForward(src, tape, f32);
    if (options.adjoints()) {
      emitReverse(src, tape, f32);
    }

    src.append("    accValue += (double) v").append(tape.outputNode()).append(";\n");
    if (options.adjoints()) {
      for (int j = 0; j < nIn; j++) {
        src.append("    accAdj").append(j).append(" += (double) d").append(tape.inputNode(j)).append(";\n");
      }
    }
    src.append("  }\n");

    int channels = options.adjoints() ? nIn + 1 : 1;
    src.append("  __shared__ double sh[BLOCK];\n");
    for (int c = 0; c < channels; c++) {
      String source = c == 0 ? "accValue" : "accAdj" + (c - 1);
      src.append("  sh[threadIdx.x] = ").append(source).append(";\n")
          .append("  __syncthreads();\n")
          .append("  for (int s = BLOCK / 2; s > 0; s >>= 1) {\n")
          .append("    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];\n")
          .append("    __syncthreads();\n")
          .append("  }\n")
          .append("  if (threadIdx.x == 0) partials[blockIdx.x * ").append(channels)
          .append(" + ").append(c).append("] = sh[0];\n")
          .append("  __syncthreads();\n");
    }
    src.append("}\n");
    return src.toString();
  }

  // ---- segment-checkpointed kernel (long tapes) -------------------------------

  /**
   * The checkpointed counterpart of {@link #generate}: same maths, but the
   * forward runs segment by segment and the reverse recomputes each segment
   * from a small boundary checkpoint instead of keeping every {@code v_i} live.
   * The kernel takes two extra parameters — the per-invocation scratch buffer
   * and the invocation count that strides it. See {@link AadCheckpointPlan}.
   */
  public static String generateCheckpointed(AadTape tape, AadOptions options, AadCheckpointPlan plan) {
    boolean f32 = options.precision() == AadOptions.Precision.FLOAT32;
    String real = f32 ? "float" : "double";
    String zero = f32 ? "0.0f" : "0.0";
    String one = f32 ? "1.0f" : "1.0";
    int nIn = tape.inputCount();
    int channels = nIn + 1;
    int outNode = tape.outputNode();
    int[] bound = plan.bound;
    int segments = plan.segments;

    StringBuilder src = new StringBuilder(1 << 16);
    src.append("#define BLOCK ").append(BLOCK).append('\n');
    src.append("typedef ").append(real).append(" real;\n");
    appendRng(src, f32);

    src.append("extern \"C\" __global__ void ").append(KERNEL_NAME).append("(\n")
        .append("    const double* __restrict__ inputs,\n")
        .append("    unsigned long long nPaths,\n")
        .append("    unsigned long long pathOffset,\n")
        .append("    unsigned long long seed,\n")
        .append("    double* __restrict__ partials,\n")
        .append("    real* __restrict__ scratch,\n")
        .append("    unsigned long long invocations) {\n");
    for (int j = 0; j < nIn; j++) {
      src.append("  const real in").append(j).append(" = (real) inputs[").append(j).append("];\n");
    }
    src.append("  double accValue = 0.0;\n");
    for (int j = 0; j < nIn; j++) {
      src.append("  double accAdj").append(j).append(" = 0.0;\n");
    }
    src.append("  const unsigned long long tid = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;\n")
        .append("  const unsigned long long stride = (unsigned long long) blockDim.x * gridDim.x;\n")
        .append("  for (unsigned long long path = tid; path < nPaths; path += stride) {\n")
        .append("    Rng rng; rng_init(rng, path + pathOffset, seed);\n")
        .append("    const unsigned long long srow = tid * ").append(plan.slotsPerPath).append("ull;\n");
    for (int i = 0; i < plan.nodes; i++) {
      if (plan.global[i]) {
        src.append("    real g_v").append(i).append(";\n");
      }
    }

    // ---- forward, segment by segment ----
    for (int s = 0; s < segments; s++) {
      src.append("    {\n");
      if (s >= 1) {
        for (int node : plan.slice[s]) {
          src.append("      real v").append(node).append(" = scratch[srow + ")
              .append(plan.slotOf(s, node)).append("ull];\n");
        }
      }
      for (int i = bound[s]; i < bound[s + 1]; i++) {
        src.append("      ").append(plan.global[i] ? "g_v" + i : "real v" + i)
            .append(" = ").append(ckFwdRhs(tape, i, f32, plan)).append(";\n");
      }
      if (s + 1 < segments) {
        for (int node : plan.slice[s + 1]) {
          src.append("      scratch[srow + ").append(plan.slotOf(s + 1, node))
              .append("ull] = v").append(node).append(";\n");
        }
      }
      if (s == segments - 1) {
        src.append("      accValue += (double) v").append(outNode).append(";\n");
      }
      src.append("    }\n");
    }

    // ---- reverse, segment by segment (recompute + adjoint) ----
    for (int i = 0; i < plan.nodes; i++) {
      if (plan.inCarry(i)) {
        src.append("    real dc_").append(i).append(" = ").append(zero).append(";\n");
      }
    }
    if (plan.inCarry(outNode)) {
      src.append("    dc_").append(outNode).append(" = ").append(one).append(";\n");
    }
    for (int s = segments - 1; s >= 0; s--) {
      src.append("    {\n");
      if (s >= 1) {
        for (int node : plan.slice[s]) {
          src.append("      real v").append(node).append(" = scratch[srow + ")
              .append(plan.slotOf(s, node)).append("ull];\n");
        }
      }
      for (int i = bound[s]; i < bound[s + 1]; i++) {
        src.append("      ").append(plan.global[i] ? "g_v" + i : "real v" + i)
            .append(" = ").append(ckFwdRhs(tape, i, f32, plan)).append(";\n");
      }
      for (int i = bound[s]; i < bound[s + 1]; i++) {
        if (!tape.isActive(i)) {
          continue;
        }
        String seed = i == outNode && !plan.inCarry(i) ? one
            : plan.inCarry(i) ? "dc_" + i : zero;
        src.append("      real d").append(i).append(" = ").append(seed).append(";\n");
      }
      for (int i = bound[s + 1] - 1; i >= bound[s]; i--) {
        if (tape.isActive(i)) {
          ckRevLine(src, tape, i, f32, plan, bound[s]);
        }
      }
      for (int j = 0; j < nIn; j++) {
        if (plan.segmentOf(tape.inputNode(j)) == s) {
          src.append("      accAdj").append(j).append(" += (double) d").append(tape.inputNode(j)).append(";\n");
        }
      }
      src.append("    }\n");
    }
    src.append("  }\n");

    src.append("  __shared__ double sh[BLOCK];\n");
    for (int c = 0; c < channels; c++) {
      String source = c == 0 ? "accValue" : "accAdj" + (c - 1);
      src.append("  sh[threadIdx.x] = ").append(source).append(";\n")
          .append("  __syncthreads();\n")
          .append("  for (int s = BLOCK / 2; s > 0; s >>= 1) {\n")
          .append("    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];\n")
          .append("    __syncthreads();\n")
          .append("  }\n")
          .append("  if (threadIdx.x == 0) partials[blockIdx.x * ").append(channels)
          .append(" + ").append(c).append("] = sh[0];\n")
          .append("  __syncthreads();\n");
    }
    src.append("}\n");
    return src.toString();
  }

  private static String ckFwdRhs(AadTape tape, int i, boolean f32, AadCheckpointPlan plan) {
    int a = tape.argA(i);
    int b = tape.argB(i);
    String va = AadCheckpointPlan.nodeArgA(tape.op(i)) ? plan.vref(a) : null;
    String vb = AadCheckpointPlan.nodeArgB(tape.op(i)) ? plan.vref(b) : null;
    String fmax = f32 ? "fmaxf" : "fmax";
    String fmin = f32 ? "fminf" : "fmin";
    return switch (tape.op(i)) {
      case CONST -> literal(tape.constant(i), f32);
      case INPUT -> "in" + a;
      case RANDN -> "rng_normal(rng, " + randArg(tape, i) + ")";
      case RANDU -> "rng_uniform(rng, " + randArg(tape, i) + ")";
      case ADD -> va + " + " + vb;
      case SUB -> va + " - " + vb;
      case MUL -> va + " * " + vb;
      case DIV -> va + " / " + vb;
      case NEG -> "-" + va;
      case EXP -> (f32 ? "__expf(" : "exp(") + va + ")";
      case LOG -> (f32 ? "logf(" : "log(") + va + ")";
      case SQRT -> (f32 ? "sqrtf(" : "sqrt(") + va + ")";
      case ABS -> (f32 ? "fabsf(" : "fabs(") + va + ")";
      case MAX -> fmax + "(" + va + ", " + vb + ")";
      case MIN -> fmin + "(" + va + ", " + vb + ")";
    };
  }

  private static void ckRevLine(StringBuilder src, AadTape tape, int i, boolean f32,
                                AadCheckpointPlan plan, int segLo) {
    int a = tape.argA(i);
    int b = tape.argB(i);
    boolean activeA = a >= 0 && tape.isActive(a);
    boolean activeB = b >= 0 && tape.isActive(b);
    String di = "d" + i;
    String da = a >= 0 ? (a >= segLo ? "d" + a : "dc_" + a) : null;
    String db = b >= 0 ? (b >= segLo ? "d" + b : "dc_" + b) : null;
    String va = a >= 0 ? plan.vref(a) : null;
    String vb = b >= 0 ? plan.vref(b) : null;
    String vi = plan.vref(i);
    String zero = f32 ? "0.0f" : "0.0";
    String half = f32 ? "0.5f" : "0.5";
    switch (tape.op(i)) {
      case CONST, INPUT, RANDN, RANDU -> {
      }
      case ADD -> {
        if (activeA) src.append("      ").append(da).append(" += ").append(di).append(";\n");
        if (activeB) src.append("      ").append(db).append(" += ").append(di).append(";\n");
      }
      case SUB -> {
        if (activeA) src.append("      ").append(da).append(" += ").append(di).append(";\n");
        if (activeB) src.append("      ").append(db).append(" -= ").append(di).append(";\n");
      }
      case MUL -> {
        if (activeA) src.append("      ").append(da).append(" += ").append(di).append(" * ").append(vb).append(";\n");
        if (activeB) src.append("      ").append(db).append(" += ").append(di).append(" * ").append(va).append(";\n");
      }
      case DIV -> {
        if (activeA) src.append("      ").append(da).append(" += ").append(di).append(" / ").append(vb).append(";\n");
        if (activeB) src.append("      ").append(db).append(" -= ").append(di).append(" * ").append(vi)
            .append(" / ").append(vb).append(";\n");
      }
      case NEG -> src.append("      ").append(da).append(" -= ").append(di).append(";\n");
      case EXP -> src.append("      ").append(da).append(" += ").append(di).append(" * ").append(vi).append(";\n");
      case LOG -> src.append("      ").append(da).append(" += ").append(di).append(" / ").append(va).append(";\n");
      case SQRT -> src.append("      ").append(da).append(" += ").append(di).append(" * ").append(half)
          .append(" / ").append(vi).append(";\n");
      case ABS -> src.append("      ").append(da).append(" += ").append(va).append(" < ").append(zero)
          .append(" ? -").append(di).append(" : ").append(di).append(";\n");
      case MAX -> {
        String taken = "(" + va + " >= " + vb + ")";
        if (activeA) src.append("      if ").append(taken).append(" ").append(da).append(" += ").append(di).append(";\n");
        if (activeB) src.append("      if (!").append(taken).append(") ").append(db).append(" += ").append(di).append(";\n");
      }
      case MIN -> {
        String taken = "(" + va + " <= " + vb + ")";
        if (activeA) src.append("      if ").append(taken).append(" ").append(da).append(" += ").append(di).append(";\n");
        if (activeB) src.append("      if (!").append(taken).append(") ").append(db).append(" += ").append(di).append(";\n");
      }
    }
  }

  private static void emitForward(StringBuilder src, AadTape tape, boolean f32) {
    String fmax = f32 ? "fmaxf" : "fmax";
    String fmin = f32 ? "fminf" : "fmin";
    for (int i = 0; i < tape.size(); i++) {
      int a = tape.argA(i);
      int b = tape.argB(i);
      src.append("    const real v").append(i).append(" = ");
      switch (tape.op(i)) {
        case CONST -> src.append(literal(tape.constant(i), f32));
        case INPUT -> src.append("in").append(a);
        case RANDN -> src.append("rng_normal(rng, ").append(randArg(tape, i)).append(")");
        case RANDU -> src.append("rng_uniform(rng, ").append(randArg(tape, i)).append(")");
        case ADD -> src.append('v').append(a).append(" + v").append(b);
        case SUB -> src.append('v').append(a).append(" - v").append(b);
        case MUL -> src.append('v').append(a).append(" * v").append(b);
        case DIV -> src.append('v').append(a).append(" / v").append(b);
        case NEG -> src.append("-v").append(a);
        case EXP -> src.append(f32 ? "__expf(v" : "exp(v").append(a).append(')');
        case LOG -> src.append(f32 ? "logf(v" : "log(v").append(a).append(')');
        case SQRT -> src.append(f32 ? "sqrtf(v" : "sqrt(v").append(a).append(')');
        case ABS -> src.append(f32 ? "fabsf(v" : "fabs(v").append(a).append(')');
        case MAX -> src.append(fmax).append("(v").append(a).append(", v").append(b).append(')');
        case MIN -> src.append(fmin).append("(v").append(a).append(", v").append(b).append(')');
      }
      src.append(";\n");
    }
  }

  private static void emitReverse(StringBuilder src, AadTape tape, boolean f32) {
    String zero = f32 ? "0.0f" : "0.0";
    String half = f32 ? "0.5f" : "0.5";
    for (int i = 0; i < tape.size(); i++) {
      if (tape.isActive(i)) {
        src.append("    real d").append(i).append(" = ").append(zero).append(";\n");
      }
    }
    src.append("    d").append(tape.outputNode()).append(" = ").append(f32 ? "1.0f" : "1.0").append(";\n");

    for (int i = tape.size() - 1; i >= 0; i--) {
      if (!tape.isActive(i)) {
        continue;
      }
      int a = tape.argA(i);
      int b = tape.argB(i);
      boolean activeA = a >= 0 && tape.isActive(a);
      boolean activeB = b >= 0 && tape.isActive(b);
      switch (tape.op(i)) {
        case CONST, INPUT, RANDN, RANDU -> {
        }
        case ADD -> {
          if (activeA) src.append("    d").append(a).append(" += d").append(i).append(";\n");
          if (activeB) src.append("    d").append(b).append(" += d").append(i).append(";\n");
        }
        case SUB -> {
          if (activeA) src.append("    d").append(a).append(" += d").append(i).append(";\n");
          if (activeB) src.append("    d").append(b).append(" -= d").append(i).append(";\n");
        }
        case MUL -> {
          if (activeA) src.append("    d").append(a).append(" += d").append(i).append(" * v").append(b).append(";\n");
          if (activeB) src.append("    d").append(b).append(" += d").append(i).append(" * v").append(a).append(";\n");
        }
        case DIV -> {
          if (activeA) src.append("    d").append(a).append(" += d").append(i).append(" / v").append(b).append(";\n");
          if (activeB) src.append("    d").append(b).append(" -= d").append(i).append(" * v").append(i)
              .append(" / v").append(b).append(";\n");
        }
        case NEG -> src.append("    d").append(a).append(" -= d").append(i).append(";\n");
        case EXP -> src.append("    d").append(a).append(" += d").append(i).append(" * v").append(i).append(";\n");
        case LOG -> src.append("    d").append(a).append(" += d").append(i).append(" / v").append(a).append(";\n");
        case SQRT -> src.append("    d").append(a).append(" += d").append(i).append(" * ").append(half)
            .append(" / v").append(i).append(";\n");
        case ABS -> src.append("    d").append(a).append(" += v").append(a).append(" < ").append(zero)
            .append(" ? -d").append(i).append(" : d").append(i).append(";\n");
        // A max/min picks one branch per scenario, so the adjoint follows the
        // branch this thread actually took rather than being fixed at record time.
        case MAX -> emitSelect(src, i, a, b, activeA, activeB, ">=");
        case MIN -> emitSelect(src, i, a, b, activeA, activeB, "<=");
      }
    }
  }

  private static void emitSelect(StringBuilder src, int i, int a, int b,
                                 boolean activeA, boolean activeB, String comparison) {
    String taken = "(v" + a + " " + comparison + " v" + b + ")";
    if (activeA) {
      src.append("    if ").append(taken).append(" d").append(a).append(" += d").append(i).append(";\n");
    }
    if (activeB) {
      src.append("    if (!").append(taken).append(") d").append(b).append(" += d").append(i).append(";\n");
    }
  }


  /** Counter argument for a RANDN/RANDU node: the per-(stream,kind) ordinal, with a
   *  stream stride folded in only when the tape has more than one stream (so a
   *  single-stream tape emits exactly what it did before this method existed). */
  private static String randArg(AadTape tape, int node) {
    int ord = tape.randOrdinal(node);
    if (tape.randStreamCount() <= 1) {
      return ord + "u";
    }
    long mix = (long) tape.randStreamOf(node) * 0x01000000L;   // even stride, stream 0 -> +0
    return "(" + ord + "u + " + mix + "u)";
  }

  private static String literal(double value, boolean f32) {
    String text = Double.toString(value);
    return f32 ? text + "f" : text;
  }

  /**
   * Philox2x32-10 keyed on the scenario index and indexed by draw number: a
   * counter-based generator, so every thread reproduces its own independent
   * stream from nothing but the path number, with no state to load. Taking the
   * draw index {@code k} as an argument (rather than an internal counter) makes
   * a draw a pure function of {@code (path, k)}, which is what lets the adjoint
   * sweep recompute a checkpointed segment's randoms without saving RNG state.
   *
   * <p>Byte-for-byte the previous stateful generator: draw {@code k} uses Philox
   * counter {@code k >> 1}; even {@code k} takes the cosine leg of the
   * Box-Muller pair, odd {@code k} the sine leg.
   */
  private static void appendRng(StringBuilder src, boolean f32) {
    String sqrt = f32 ? "sqrtf" : "sqrt";
    String log = f32 ? "logf" : "log";
    String sincos = f32 ? "__sincosf" : "sincos";
    src.append("""
        struct Rng { unsigned int lo; unsigned int hi; };

        __device__ __forceinline__ void rng_init(Rng &g, unsigned long long path, unsigned long long seed) {
          g.lo = (unsigned int) (path & 0xffffffffull) ^ (unsigned int) (seed & 0xffffffffull);
          g.hi = (unsigned int) (path >> 32) ^ (unsigned int) (seed >> 32);
        }

        __device__ __forceinline__ real rng_normal(Rng &g, unsigned int k) {
          unsigned int c0 = g.lo;
          unsigned int c1 = g.hi ^ (k >> 1);
          unsigned int key = 0x1BD11BDAu;
          #pragma unroll
          for (int i = 0; i < 10; i++) {
            unsigned int hi = __umulhi(0xD256D193u, c0);
            unsigned int lo = 0xD256D193u * c0;
            c0 = hi ^ key ^ c1;
            c1 = lo;
            key += 0x9E3779B9u;
          }
          real u1 = ((real) c0 + (real) 0.5) * (real) 2.3283064365386963e-10;
          real u2 = ((real) c1 + (real) 0.5) * (real) 2.3283064365386963e-10;
          real radius = RSQRT(-(real) 2.0 * RLOG(u1));
          real s, c;
          RSINCOS((real) 6.283185307179586 * u2, &s, &c);
          return (k & 1u) == 0u ? radius * c : radius * s;
        }

        __device__ __forceinline__ real rng_uniform(Rng &g, unsigned int k) {
          unsigned int c0 = g.lo;
          unsigned int c1 = g.hi ^ k ^ 0x9E3779B9u;
          unsigned int key = 0x1BD11BDAu;
          #pragma unroll
          for (int i = 0; i < 10; i++) {
            unsigned int hi = __umulhi(0xD256D193u, c0);
            unsigned int lo = 0xD256D193u * c0;
            c0 = hi ^ key ^ c1;
            c1 = lo;
            key += 0x9E3779B9u;
          }
          return ((real) c0 + (real) 0.5) * (real) 2.3283064365386963e-10;
        }
        """
        .replace("RSQRT", sqrt)
        .replace("RLOG", log)
        .replace("RSINCOS", sincos));
  }
}
