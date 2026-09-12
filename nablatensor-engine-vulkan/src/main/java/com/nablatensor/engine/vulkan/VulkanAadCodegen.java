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

import com.nablatensor.engine.AadOp;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadTape;

import java.util.Arrays;

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
 *
 * <p>Two things exploit the draws being a pure function of {@code (path, k)}
 * rather than of the market. A dispatch can price several input sets
 * ({@link Config#markets()}) from one generation of each path's draws, held in
 * registers — the GPU form of a draw cache for a revaluation ladder. And a long
 * tape's adjoint sweep can be checkpointed ({@link Config#segments()}): the
 * forward values are dropped after the forward sweep and recomputed segment by
 * segment, draws included, instead of being kept live and spilled to scratch
 * memory. Both leave every result bit-identical to the plain kernel.
 */
final class VulkanAadCodegen {

  /** Workgroup size; matches the CUDA engine's block. */
  static final int LOCAL_SIZE = 256;
  /** Upper bound on scenarios handled by a single {@code vkCmdDispatch}. */
  static final int MAX_DISPATCH_PATHS = 1 << 27;

  /**
   * Tapes at least this long have their adjoint sweep checkpointed unless
   * {@code -Dnablatensor.vulkan.ckpt} says otherwise. Below it the kernel's
   * working set fits the register file and a second forward sweep only costs;
   * above it the plain kernel spills and slows by several times.
   */
  static final int CHECKPOINT_NODES = Integer.getInteger("nablatensor.vulkan.ckpt.nodes", 768);
  static final int CHECKPOINT_SEGMENTS = 16;

  /**
   * @param markets  input sets priced per path from the same draws in one dispatch
   * @param segments pieces the adjoint sweep is checkpointed into; {@code 1} keeps every forward value live
   */
  record Config(int markets, int segments) {

    /** One market; checkpointing from the {@code nablatensor.vulkan.ckpt} property or the node threshold. */
    static Config forTape(AadTape tape, AadOptions options) {
      int segments = 1;
      if (options.adjoints()) {
        Integer explicit = Integer.getInteger("nablatensor.vulkan.ckpt");
        segments = explicit != null ? Math.max(1, explicit)
            : tape.size() >= CHECKPOINT_NODES ? CHECKPOINT_SEGMENTS : 1;
      }
      return new Config(1, segments);
    }

    Config withMarkets(int m) {
      return new Config(m, segments);
    }
  }

  /** Push-constant slots, as {@code uint}s. */
  static final int PC_N_LOCAL = 0;
  static final int PC_OFF_LO = 1;
  static final int PC_OFF_HI = 2;
  static final int PC_SEED_LO = 3;
  static final int PC_SEED_HI = 4;
  /** Always zero; XORed into recomputed values so the shader compiler cannot fold them back into the originals. */
  static final int PC_ZERO = 5;
  static final int PC_COUNT = 6;

  private VulkanAadCodegen() {
  }

  static String generate(AadTape tape, AadOptions options, Config cfg) {
    boolean adjoints = options.adjoints();
    int nIn = tape.inputCount();
    int channels = adjoints ? nIn + 1 : 1;
    int markets = cfg.markets();
    boolean checkpointed = adjoints && cfg.segments() > 1;

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
        .append("  uint zero;\n")
        .append("} pc;\n");
    src.append("const uint CHANNELS = ").append(channels).append("u;\n");
    src.append("const uint MARKETS = ").append(markets).append("u;\n");
    appendRng(src);
    if (checkpointed) {
      // Bit-exact identity the compiler cannot see through (pc.zero is 0 at run time).
      src.append("float opq(float x) { return uintBitsToFloat(floatBitsToUint(x) ^ pc.zero); }\n");
    }

    src.append("shared float sh[").append(LOCAL_SIZE).append("];\n\n");
    src.append("void main() {\n");
    src.append("  uint lid = gl_LocalInvocationID.x;\n");
    src.append("  uint stride = gl_NumWorkGroups.x * ").append(LOCAL_SIZE).append("u;\n");

    for (int m = 0; m < markets; m++) {
      for (int j = 0; j < nIn; j++) {
        src.append("  float in").append(j).append(sfx(m, markets))
            .append(" = inp[").append(m * nIn + j).append("];\n");
      }
    }
    // Per-invocation sums use Kahan compensation: the scenarios one invocation
    // strides over can number in the thousands, and a naive fp32 running sum
    // loses the low bits of every later term once the partial sum outgrows a
    // single term. The compensation term carries those bits forward.
    for (int m = 0; m < markets; m++) {
      String s = sfx(m, markets);
      src.append("  float accValue").append(s).append(" = 0.0; float cValue").append(s).append(" = 0.0;\n");
      if (adjoints) {
        for (int j = 0; j < nIn; j++) {
          src.append("  float accAdj").append(j).append(s).append(" = 0.0; float cAdj").append(j).append(s).append(" = 0.0;\n");
        }
      }
    }

    src.append("  for (uint p = gl_GlobalInvocationID.x; p < pc.nLocal; p += stride) {\n");
    src.append("    uint pathLo = pc.offLo + p;\n");
    src.append("    uint carry = pathLo < pc.offLo ? 1u : 0u;\n");
    src.append("    uint pathHi = pc.offHi + carry;\n");
    src.append("    rng_init(pathLo ^ pc.seedLo, pathHi ^ pc.seedHi);\n");

    if (checkpointed) {
      emitCheckpointed(src, tape, cfg);
    } else {
      for (int i = 0; i < tape.size(); i++) {
        emitNode(src, tape, cfg, i, true);
      }
      if (adjoints) {
        emitReverse(src, tape, cfg);
      }
    }

    for (int m = 0; m < markets; m++) {
      String s = sfx(m, markets);
      kahanAdd(src, "accValue" + s, "cValue" + s, vn(tape, tape.outputNode(), m, markets));
      if (adjoints) {
        for (int j = 0; j < nIn; j++) {
          kahanAdd(src, "accAdj" + j + s, "cAdj" + j + s, dn(tape.inputNode(j), m, markets));
        }
      }
    }
    src.append("  }\n");

    for (int m = 0; m < markets; m++) {
      String s = sfx(m, markets);
      for (int c = 0; c < channels; c++) {
        String source = c == 0 ? "accValue" + s : "accAdj" + (c - 1) + s;
        int slot = m * channels + c;
        src.append("  sh[lid] = ").append(source).append(";\n")
            .append("  barrier();\n")
            .append("  for (uint s = ").append(LOCAL_SIZE / 2).append("u; s > 0u; s >>= 1) {\n")
            .append("    if (lid < s) sh[lid] += sh[lid + s];\n")
            .append("    barrier();\n")
            .append("  }\n")
            .append("  if (lid == 0u) partials[gl_WorkGroupID.x * (CHANNELS * MARKETS) + ").append(slot).append("u] = sh[0];\n")
            .append("  barrier();\n");
      }
    }
    src.append("}\n");
    return src.toString();
  }

  private static String sfx(int m, int markets) {
    return markets == 1 ? "" : "_" + m;
  }

  private static boolean marketFree(AadOp op) {
    return op == AadOp.RANDN || op == AadOp.CONST;
  }

  /**
   * Name of node {@code i}'s forward value for market {@code m}. Draws and
   * constants do not depend on the market, so they carry no suffix and are
   * computed once however many markets share the dispatch.
   */
  private static String vn(AadTape tape, int i, int m, int markets) {
    return marketFree(tape.op(i)) ? "v" + i : "v" + i + sfx(m, markets);
  }

  private static String dn(int i, int m, int markets) {
    return "d" + i + sfx(m, markets);
  }

  /** {@code acc += rhs} carried in single precision with a Kahan compensation term. */
  private static void kahanAdd(StringBuilder src, String acc, String comp, String rhs) {
    src.append("    { float y = ").append(rhs).append(" - ").append(comp).append(";")
        .append(" float t = ").append(acc).append(" + y;")
        .append(' ').append(comp).append(" = (t - ").append(acc).append(") - y;")
        .append(' ').append(acc).append(" = t; }\n");
  }

  /**
   * One forward node for every market, declared ({@code float v<i>}) or
   * assigned to an already-declared variable.
   */
  private static void emitNode(StringBuilder src, AadTape tape, Config cfg, int i, boolean declare) {
    int markets = cfg.markets();
    int a = tape.argA(i);
    int b = tape.argB(i);
    AadOp op = tape.op(i);
    String decl = declare ? "float " : "";
    if (marketFree(op)) {
      src.append("    ").append(decl).append("v").append(i).append(" = ")
          .append(op == AadOp.RANDN ? "rng_normal(" + a + "u)" : literal(tape.constant(i)))
          .append(";\n");
      return;
    }
    for (int m = 0; m < markets; m++) {
      String va = a >= 0 ? vn(tape, a, m, markets) : "";
      String vb = b >= 0 ? vn(tape, b, m, markets) : "";
      src.append("    ").append(decl).append(vn(tape, i, m, markets)).append(" = ");
      switch (op) {
        case CONST, RANDN -> throw new IllegalStateException();
        case INPUT -> src.append("in").append(a).append(sfx(m, markets));
        case ADD -> src.append(va).append(" + ").append(vb);
        case SUB -> src.append(va).append(" - ").append(vb);
        case MUL -> src.append(va).append(" * ").append(vb);
        case DIV -> src.append(va).append(" / ").append(vb);
        case NEG -> src.append('-').append(va);
        case EXP -> src.append("exp(").append(va).append(')');
        case LOG -> src.append("log(").append(va).append(')');
        case SQRT -> src.append("sqrt(").append(va).append(')');
        case ABS -> src.append("abs(").append(va).append(')');
        case MAX -> src.append("max(").append(va).append(", ").append(vb).append(')');
        case MIN -> src.append("min(").append(va).append(", ").append(vb).append(')');
      }
      src.append(";\n");
    }
  }

  private static void emitReverse(StringBuilder src, AadTape tape, Config cfg) {
    int markets = cfg.markets();
    for (int m = 0; m < markets; m++) {
      for (int i = 0; i < tape.size(); i++) {
        if (tape.isActive(i)) {
          src.append("    float ").append(dn(i, m, markets)).append(" = 0.0;\n");
        }
      }
      src.append("    ").append(dn(tape.outputNode(), m, markets)).append(" = 1.0;\n");
    }
    emitReverseRange(src, tape, cfg, 0, tape.size());
  }

  /** The adjoint rules of nodes {@code [from, to)}, highest first, for every market. */
  private static void emitReverseRange(StringBuilder src, AadTape tape, Config cfg, int from, int to) {
    int markets = cfg.markets();
    for (int i = to - 1; i >= from; i--) {
      if (!tape.isActive(i)) {
        continue;
      }
      int a = tape.argA(i);
      int b = tape.argB(i);
      boolean activeA = a >= 0 && tape.isActive(a);
      boolean activeB = b >= 0 && tape.isActive(b);
      for (int m = 0; m < markets; m++) {
        String di = dn(i, m, markets);
        String da = dn(a, m, markets);
        String db = dn(b, m, markets);
        String vi = vn(tape, i, m, markets);
        String va = a >= 0 ? vn(tape, a, m, markets) : "";
        String vb = b >= 0 ? vn(tape, b, m, markets) : "";
        switch (tape.op(i)) {
          case CONST, INPUT, RANDN -> {
          }
          case ADD -> {
            if (activeA) src.append("    ").append(da).append(" += ").append(di).append(";\n");
            if (activeB) src.append("    ").append(db).append(" += ").append(di).append(";\n");
          }
          case SUB -> {
            if (activeA) src.append("    ").append(da).append(" += ").append(di).append(";\n");
            if (activeB) src.append("    ").append(db).append(" -= ").append(di).append(";\n");
          }
          case MUL -> {
            if (activeA) src.append("    ").append(da).append(" += ").append(di).append(" * ").append(vb).append(";\n");
            if (activeB) src.append("    ").append(db).append(" += ").append(di).append(" * ").append(va).append(";\n");
          }
          case DIV -> {
            if (activeA) src.append("    ").append(da).append(" += ").append(di).append(" / ").append(vb).append(";\n");
            if (activeB) src.append("    ").append(db).append(" -= ").append(di).append(" * ").append(vi)
                .append(" / ").append(vb).append(";\n");
          }
          case NEG -> src.append("    ").append(da).append(" -= ").append(di).append(";\n");
          case EXP -> src.append("    ").append(da).append(" += ").append(di).append(" * ").append(vi).append(";\n");
          case LOG -> src.append("    ").append(da).append(" += ").append(di).append(" / ").append(va).append(";\n");
          case SQRT -> src.append("    ").append(da).append(" += ").append(di).append(" * 0.5 / ").append(vi).append(";\n");
          case ABS -> src.append("    ").append(da).append(" += ").append(va).append(" < 0.0 ? -").append(di)
              .append(" : ").append(di).append(";\n");
          // A max/min picks one branch per scenario, so the adjoint follows the
          // branch this invocation actually took rather than being fixed at record time.
          case MAX -> emitSelect(src, di, da, db, va, vb, activeA, activeB, ">=");
          case MIN -> emitSelect(src, di, da, db, va, vb, activeA, activeB, "<=");
        }
      }
    }
  }

  private static void emitSelect(StringBuilder src, String di, String da, String db, String va, String vb,
                                 boolean activeA, boolean activeB, String comparison) {
    String taken = "(" + va + " " + comparison + " " + vb + ")";
    if (activeA) {
      src.append("    if ").append(taken).append(' ').append(da).append(" += ").append(di).append(";\n");
    }
    if (activeB) {
      src.append("    if (!").append(taken).append(") ").append(db).append(" += ").append(di).append(";\n");
    }
  }

  // ---- checkpointed adjoint ------------------------------------------------

  private static boolean refsA(AadOp op) {
    return switch (op) {
      case CONST, INPUT, RANDN -> false;
      default -> true;
    };
  }

  private static boolean refsB(AadOp op) {
    return switch (op) {
      case ADD, SUB, MUL, DIV, MAX, MIN -> true;
      default -> false;
    };
  }

  /**
   * The forward and adjoint sweeps with the tape cut into {@code segments}
   * pieces at the positions where the fewest values cross. Only those crossing
   * values (and the inputs and output) live for the whole kernel; everything
   * else is computed once in the forward sweep, dropped, and recomputed in the
   * adjoint sweep from the checkpoint — the draws included, since a draw is a
   * pure function of {@code (path, k)}. That trades a second forward sweep
   * for an adjoint whose working set fits in registers instead of spilling
   * the whole tape to scratch memory.
   */
  private static void emitCheckpointed(StringBuilder src, AadTape tape, Config cfg) {
    int n = tape.size();
    int markets = cfg.markets();
    int[] lastUse = new int[n];
    Arrays.fill(lastUse, -1);
    for (int i = 0; i < n; i++) {
      AadOp op = tape.op(i);
      if (refsA(op)) lastUse[tape.argA(i)] = i;
      if (refsB(op)) lastUse[tape.argB(i)] = i;
    }
    // cut c sits where the fewest nodes before it are still read at or after it
    int segments = Math.min(cfg.segments(), n);
    int[] cuts = new int[segments + 1];
    cuts[segments] = n;
    for (int s = 1; s < segments; s++) {
      int target = (int) ((long) s * n / segments);
      int window = Math.max(1, n / (4 * segments));
      int best = target;
      int bestSize = Integer.MAX_VALUE;
      for (int c = Math.max(cuts[s - 1] + 1, target - window); c <= Math.min(n - 1, target + window); c++) {
        int size = 0;
        for (int j = 0; j < c; j++) {
          if (lastUse[j] >= c) size++;
        }
        if (size < bestSize || (size == bestSize && Math.abs(c - target) < Math.abs(best - target))) {
          bestSize = size;
          best = c;
        }
      }
      cuts[s] = best;
    }
    boolean[] persistent = new boolean[n];
    for (int s = 1; s < segments; s++) {
      int c = cuts[s];
      for (int j = 0; j < c; j++) {
        if (lastUse[j] >= c) persistent[j] = true;
      }
    }
    persistent[tape.outputNode()] = true;
    for (int j = 0; j < tape.inputCount(); j++) {
      persistent[tape.inputNode(j)] = true;
    }
    for (int i = 0; i < n; i++) {
      if (persistent[i]) declareAll(src, tape, cfg, "v", i, null);
    }

    // forward sweep: segment-local values die at the block's end
    for (int s = 0; s < segments; s++) {
      src.append("    {\n");
      for (int i = cuts[s]; i < cuts[s + 1]; i++) {
        emitNode(src, tape, cfg, i, !persistent[i]);
      }
      src.append("    }\n");
    }

    // adjoint sweep, last segment first, recomputing each segment's locals
    for (int i = 0; i < n; i++) {
      if (persistent[i] && tape.isActive(i)) declareAll(src, tape, cfg, "d", i, "0.0");
    }
    for (int m = 0; m < markets; m++) {
      src.append("    ").append(dn(tape.outputNode(), m, markets)).append(" = 1.0;\n");
    }
    for (int s = segments - 1; s >= 0; s--) {
      src.append("    {\n");
      // Recompute from opaque copies of the checkpoint and an opaque re-key of
      // the generator; otherwise common-subexpression elimination merges the
      // recomputation with the forward sweep and keeps every value live again.
      src.append("    rng_init((pathLo ^ pc.seedLo) ^ pc.zero, (pathHi ^ pc.seedHi) ^ pc.zero);\n");
      for (int j = 0; j < cuts[s]; j++) {
        if (lastUse[j] >= cuts[s]) {
          int copies = marketFree(tape.op(j)) ? 1 : markets;
          for (int m = 0; m < copies; m++) {
            String name = vn(tape, j, m, markets);
            src.append("    float ").append(name).append(" = opq(").append(name).append(");\n");
          }
        }
      }
      for (int i = cuts[s]; i < cuts[s + 1]; i++) {
        if (!persistent[i]) emitNode(src, tape, cfg, i, true);
      }
      for (int i = cuts[s]; i < cuts[s + 1]; i++) {
        if (!persistent[i] && tape.isActive(i)) declareAll(src, tape, cfg, "d", i, "0.0");
      }
      emitReverseRange(src, tape, cfg, cuts[s], cuts[s + 1]);
      src.append("    }\n");
    }
  }

  /** Declares {@code <prefix><i>} for every market (once, if the node is market-free), optionally initialised. */
  private static void declareAll(StringBuilder src, AadTape tape, Config cfg, String prefix, int i, String init) {
    int markets = cfg.markets();
    boolean value = prefix.equals("v");
    int copies = value && marketFree(tape.op(i)) ? 1 : markets;
    for (int m = 0; m < copies; m++) {
      src.append("    float ").append(value ? vn(tape, i, m, markets) : dn(i, m, markets));
      if (init != null) src.append(" = ").append(init);
      src.append(";\n");
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
