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
package com.nablatensor.engine.vulkan;

import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadTape;

/**
 * Translates a recorded {@link AadTape} into a GLSL {@code compute} shader that
 * runs one scenario per invocation: a fully unrolled forward sweep followed by
 * the adjoint sweep, both in registers, with the per-scenario normal stream
 * generated in-invocation by a counter-based Philox so a replay touches no
 * memory beyond the handful of inputs and the per-workgroup reduction.
 *
 * <p>This is the line-for-line GLSL counterpart of {@code CudaAadCodegen}. The
 * differences are only those forced by the target: single precision throughout
 * (an APU runs fp64 at a small fraction of fp32, and the Vulkan runtime's host
 * copies are float-typed), 64-bit path arithmetic synthesised from {@code uint}
 * pairs because core GLSL has no 64-bit integer, and a {@code shared}-array
 * reduction in place of {@code __shared__}.
 */
final class VulkanAadCodegen {

  /** Workgroup size; matches the CUDA engine's block. */
  static final int LOCAL_SIZE = 256;
  /** Upper bound on scenarios handled by a single {@code vkCmdDispatch}. */
  static final int MAX_DISPATCH_PATHS = 1 << 27;

  private VulkanAadCodegen() {
  }

  static String generate(AadTape tape, AadOptions options) {
    boolean adjoints = options.adjoints();
    int nIn = tape.inputCount();
    int channels = adjoints ? nIn + 1 : 1;

    StringBuilder src = new StringBuilder(1 << 16);
    src.append("#version 450\n");
    src.append("layout(local_size_x = ").append(LOCAL_SIZE).append(") in;\n");
    src.append("layout(std430, binding = 0) readonly buffer Inputs { float inp[]; };\n");
    src.append("layout(std430, binding = 1) writeonly buffer Partials { float partials[]; };\n");
    src.append("layout(push_constant) uniform Push {\n")
        .append("  uint nLocal;\n")        // scenarios in this dispatch
        .append("  uint offLo;\n")         // 64-bit path offset, low word
        .append("  uint offHi;\n")         // 64-bit path offset, high word
        .append("  uint seedLo;\n")
        .append("  uint seedHi;\n")
        .append("} pc;\n");
    src.append("const uint CHANNELS = ").append(channels).append("u;\n");
    appendRng(src);

    src.append("shared float sh[").append(LOCAL_SIZE).append("];\n\n");
    src.append("void main() {\n");
    src.append("  uint lid = gl_LocalInvocationID.x;\n");
    src.append("  uint stride = gl_NumWorkGroups.x * ").append(LOCAL_SIZE).append("u;\n");

    for (int j = 0; j < nIn; j++) {
      src.append("  float in").append(j).append(" = inp[").append(j).append("];\n");
    }
    // Per-invocation sums use Kahan compensation: the scenarios one invocation
    // strides over can number in the thousands, and a naive fp32 running sum
    // loses the low bits of every later term once the partial sum outgrows a
    // single term. The compensation term carries those bits forward.
    src.append("  float accValue = 0.0; float cValue = 0.0;\n");
    if (adjoints) {
      for (int j = 0; j < nIn; j++) {
        src.append("  float accAdj").append(j).append(" = 0.0; float cAdj").append(j).append(" = 0.0;\n");
      }
    }

    src.append("  for (uint p = gl_GlobalInvocationID.x; p < pc.nLocal; p += stride) {\n");
    src.append("    uint pathLo = pc.offLo + p;\n");
    src.append("    uint carry = pathLo < pc.offLo ? 1u : 0u;\n");
    src.append("    uint pathHi = pc.offHi + carry;\n");
    src.append("    rng_init(pathLo ^ pc.seedLo, pathHi ^ pc.seedHi);\n");

    emitForward(src, tape);
    if (adjoints) {
      emitReverse(src, tape);
    }

    kahanAdd(src, "accValue", "cValue", "v" + tape.outputNode());
    if (adjoints) {
      for (int j = 0; j < nIn; j++) {
        kahanAdd(src, "accAdj" + j, "cAdj" + j, "d" + tape.inputNode(j));
      }
    }
    src.append("  }\n");

    for (int c = 0; c < channels; c++) {
      String source = c == 0 ? "accValue" : "accAdj" + (c - 1);
      src.append("  sh[lid] = ").append(source).append(";\n")
          .append("  barrier();\n")
          .append("  for (uint s = ").append(LOCAL_SIZE / 2).append("u; s > 0u; s >>= 1) {\n")
          .append("    if (lid < s) sh[lid] += sh[lid + s];\n")
          .append("    barrier();\n")
          .append("  }\n")
          .append("  if (lid == 0u) partials[gl_WorkGroupID.x * CHANNELS + ").append(c).append("u] = sh[0];\n")
          .append("  barrier();\n");
    }
    src.append("}\n");
    return src.toString();
  }

  /** {@code acc += rhs} carried in single precision with a Kahan compensation term. */
  private static void kahanAdd(StringBuilder src, String acc, String comp, String rhs) {
    src.append("    { float y = ").append(rhs).append(" - ").append(comp).append(";")
        .append(" float t = ").append(acc).append(" + y;")
        .append(' ').append(comp).append(" = (t - ").append(acc).append(") - y;")
        .append(' ').append(acc).append(" = t; }\n");
  }

  private static void emitForward(StringBuilder src, AadTape tape) {
    for (int i = 0; i < tape.size(); i++) {
      int a = tape.argA(i);
      int b = tape.argB(i);
      src.append("    float v").append(i).append(" = ");
      switch (tape.op(i)) {
        case CONST -> src.append(literal(tape.constant(i)));
        case INPUT -> src.append("in").append(a);
        case RANDN -> src.append("rng_normal(").append(a).append("u)");
        case ADD -> src.append('v').append(a).append(" + v").append(b);
        case SUB -> src.append('v').append(a).append(" - v").append(b);
        case MUL -> src.append('v').append(a).append(" * v").append(b);
        case DIV -> src.append('v').append(a).append(" / v").append(b);
        case NEG -> src.append("-v").append(a);
        case EXP -> src.append("exp(v").append(a).append(')');
        case LOG -> src.append("log(v").append(a).append(')');
        case SQRT -> src.append("sqrt(v").append(a).append(')');
        case ABS -> src.append("abs(v").append(a).append(')');
        case MAX -> src.append("max(v").append(a).append(", v").append(b).append(')');
        case MIN -> src.append("min(v").append(a).append(", v").append(b).append(')');
      }
      src.append(";\n");
    }
  }

  private static void emitReverse(StringBuilder src, AadTape tape) {
    for (int i = 0; i < tape.size(); i++) {
      if (tape.isActive(i)) {
        src.append("    float d").append(i).append(" = 0.0;\n");
      }
    }
    src.append("    d").append(tape.outputNode()).append(" = 1.0;\n");

    for (int i = tape.size() - 1; i >= 0; i--) {
      if (!tape.isActive(i)) {
        continue;
      }
      int a = tape.argA(i);
      int b = tape.argB(i);
      boolean activeA = a >= 0 && tape.isActive(a);
      boolean activeB = b >= 0 && tape.isActive(b);
      switch (tape.op(i)) {
        case CONST, INPUT, RANDN -> {
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
        case SQRT -> src.append("    d").append(a).append(" += d").append(i).append(" * 0.5 / v").append(i).append(";\n");
        case ABS -> src.append("    d").append(a).append(" += v").append(a).append(" < 0.0 ? -d").append(i)
            .append(" : d").append(i).append(";\n");
        // A max/min picks one branch per scenario, so the adjoint follows the
        // branch this invocation actually took rather than being fixed at record time.
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

  /** A finite single-precision GLSL literal; always parenthesised so a leading minus is safe. */
  private static String literal(double value) {
    float f = (float) value;
    if (Float.isNaN(f) || Float.isInfinite(f)) {
      throw new IllegalArgumentException("tape constant is not finite: " + value);
    }
    String text = Float.toString(f);
    if (text.indexOf('.') < 0 && text.indexOf('e') < 0 && text.indexOf('E') < 0) {
      text = text + ".0";
    }
    return "float(" + text + ")";
  }

  /**
   * Philox2x32-10 keyed on the scenario index and indexed by draw number,
   * byte-for-byte the generator emitted into the CUDA kernel. Taking the draw
   * index {@code k} as an argument (instead of an internal counter) makes a
   * draw a pure function of {@code (path, k)}, so the adjoint sweep can
   * recompute a checkpointed segment's randoms without saving RNG state. Draw
   * {@code k} uses Philox counter {@code k >> 1}; even {@code k} takes the
   * cosine leg of the Box-Muller pair, odd {@code k} the sine leg.
   */
  private static void appendRng(StringBuilder src) {
    src.append("""
        uint g_lo; uint g_hi;

        void rng_init(uint keyedLo, uint keyedHi) {
          g_lo = keyedLo;
          g_hi = keyedHi;
        }

        float rng_normal(uint k) {
          uint c0 = g_lo;
          uint c1 = g_hi ^ (k >> 1u);
          uint key = 0x1BD11BDAu;
          for (int i = 0; i < 10; i++) {
            uint hi; uint lo;
            umulExtended(0xD256D193u, c0, hi, lo);
            c0 = hi ^ key ^ c1;
            c1 = lo;
            key += 0x9E3779B9u;
          }
          float u1 = (float(c0) + 0.5) * 2.3283064365386963e-10;
          float u2 = (float(c1) + 0.5) * 2.3283064365386963e-10;
          float radius = sqrt(-2.0 * log(u1));
          float angle = 6.283185307179586 * u2;
          return (k & 1u) == 0u ? radius * cos(angle) : radius * sin(angle);
        }
        """);
  }
}
