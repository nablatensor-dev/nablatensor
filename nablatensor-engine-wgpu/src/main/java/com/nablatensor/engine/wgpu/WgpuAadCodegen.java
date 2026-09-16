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
package com.nablatensor.engine.wgpu;

import com.nablatensor.engine.AadOp;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadTape;

import java.util.Arrays;

/**
 * Translates a recorded {@link AadTape} into a WGSL {@code @compute} shader
 * that runs one scenario per invocation: a fully unrolled forward sweep
 * followed by the adjoint sweep, both in registers, with the per-scenario
 * normal stream generated in-invocation by a counter-based Philox so a replay
 * touches no memory beyond the handful of inputs and the per-workgroup
 * reduction.
 *
 * <p>This is the line-for-line WGSL counterpart of {@code VulkanAadCodegen}
 * (itself the GLSL counterpart of {@code CudaAadCodegen}); the segment-cut and
 * checkpointing logic is copied verbatim since it is language-agnostic, only
 * the leaf text-emitting helpers differ. Three things are forced by the
 * target rather than chosen: WGSL has no ternary operator, so a branch that
 * picks a value (rather than executing a statement) becomes {@code select};
 * WGSL has no push constants, so the 6-word push block becomes a small
 * read-only storage buffer at binding 2, indexed instead of named; and WGSL
 * has no {@code umulExtended}, so the Philox multiply is the classic
 * split-into-halves software emulation, returned as a small struct.
 */
final class WgpuAadCodegen {

  /** Workgroup size; matches the CUDA/Vulkan engines' block. */
  static final int LOCAL_SIZE = 256;
  /** Upper bound on scenarios handled by a single dispatch. */
  static final int MAX_DISPATCH_PATHS = 1 << 27;

  /**
   * Tapes at least this long have their adjoint sweep checkpointed unless
   * {@code -Dnablatensor.wgpu.ckpt} says otherwise.
   */
  static final int CHECKPOINT_NODES = Integer.getInteger("nablatensor.wgpu.ckpt.nodes", 768);
  static final int CHECKPOINT_SEGMENTS = 16;

  /**
   * @param markets  input sets priced per path from the same draws in one dispatch
   * @param segments pieces the adjoint sweep is checkpointed into; {@code 1} keeps every forward value live
   */
  record Config(int markets, int segments) {

    static Config forTape(AadTape tape, AadOptions options) {
      int segments = 1;
      if (options.adjoints()) {
        Integer explicit = Integer.getInteger("nablatensor.wgpu.ckpt");
        segments = explicit != null ? Math.max(1, explicit)
            : tape.size() >= CHECKPOINT_NODES ? CHECKPOINT_SEGMENTS : 1;
      }
      return new Config(1, segments);
    }

    Config withMarkets(int m) {
      return new Config(m, segments);
    }
  }

  /** Push-buffer word indices (see {@code array<u32, 6>} at binding 2). */
  static final int PC_N_LOCAL = 0;
  static final int PC_OFF_LO = 1;
  static final int PC_OFF_HI = 2;
  static final int PC_SEED_LO = 3;
  static final int PC_SEED_HI = 4;
  /** Always zero; XORed into recomputed values so the compiler cannot fold them back into the originals. */
  static final int PC_ZERO = 5;
  static final int PC_COUNT = 6;

  private WgpuAadCodegen() {
  }

  static String generate(AadTape tape, AadOptions options, Config cfg) {
    boolean adjoints = options.adjoints();
    int nIn = tape.inputCount();
    int channels = adjoints ? nIn + 1 : 1;
    int markets = cfg.markets();
    boolean checkpointed = adjoints && cfg.segments() > 1;

    StringBuilder src = new StringBuilder(1 << 16);
    src.append("@group(0) @binding(0) var<storage, read> inp: array<f32>;\n");
    src.append("@group(0) @binding(1) var<storage, read_write> partials: array<f32>;\n");
    src.append("@group(0) @binding(2) var<storage, read> pc: array<u32, ").append(PC_COUNT).append(">;\n");
    src.append("const CHANNELS: u32 = ").append(channels).append("u;\n");
    src.append("const MARKETS: u32 = ").append(markets).append("u;\n");
    appendRng(src);
    if (checkpointed) {
      // Bit-exact identity the compiler cannot see through (pc[5] is 0 at run time).
      src.append("fn opq(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) ^ pc[").append(PC_ZERO).append("u]); }\n");
    }

    src.append("var<workgroup> sh: array<f32, ").append(LOCAL_SIZE).append(">;\n\n");
    src.append("@compute @workgroup_size(").append(LOCAL_SIZE).append(")\n");
    src.append("fn main(\n")
        .append("  @builtin(local_invocation_id) local_id: vec3<u32>,\n")
        .append("  @builtin(global_invocation_id) global_id: vec3<u32>,\n")
        .append("  @builtin(num_workgroups) num_wg: vec3<u32>,\n")
        .append("  @builtin(workgroup_id) wg_id: vec3<u32>,\n")
        .append(") {\n");
    src.append("  let lid = local_id.x;\n");
    src.append("  let stride = num_wg.x * ").append(LOCAL_SIZE).append("u;\n");

    for (int m = 0; m < markets; m++) {
      for (int j = 0; j < nIn; j++) {
        src.append("  let in").append(j).append(sfx(m, markets))
            .append(" = inp[").append(m * nIn + j).append("];\n");
      }
    }
    // Per-invocation sums use Kahan compensation: the scenarios one invocation
    // strides over can number in the thousands, and a naive fp32 running sum
    // loses the low bits of every later term once the partial sum outgrows a
    // single term. The compensation term carries those bits forward.
    for (int m = 0; m < markets; m++) {
      String s = sfx(m, markets);
      src.append("  var accValue").append(s).append(" : f32 = 0.0; var cValue").append(s).append(" : f32 = 0.0;\n");
      if (adjoints) {
        for (int j = 0; j < nIn; j++) {
          src.append("  var accAdj").append(j).append(s).append(" : f32 = 0.0; var cAdj").append(j).append(s).append(" : f32 = 0.0;\n");
        }
      }
    }

    src.append("  for (var p = global_id.x; p < pc[").append(PC_N_LOCAL).append("u]; p += stride) {\n");
    src.append("    let pathLo = pc[").append(PC_OFF_LO).append("u] + p;\n");
    src.append("    let carry = select(0u, 1u, pathLo < pc[").append(PC_OFF_LO).append("u]);\n");
    src.append("    let pathHi = pc[").append(PC_OFF_HI).append("u] + carry;\n");
    src.append("    rng_init(pathLo ^ pc[").append(PC_SEED_LO).append("u], pathHi ^ pc[").append(PC_SEED_HI).append("u]);\n");

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
            .append("  workgroupBarrier();\n")
            .append("  for (var rs: u32 = ").append(LOCAL_SIZE / 2).append("u; rs > 0u; rs = rs >> 1u) {\n")
            .append("    if (lid < rs) { sh[lid] = sh[lid] + sh[lid + rs]; }\n")
            .append("    workgroupBarrier();\n")
            .append("  }\n")
            .append("  if (lid == 0u) { partials[wg_id.x * (CHANNELS * MARKETS) + ").append(slot).append("u] = sh[0]; }\n")
            .append("  workgroupBarrier();\n");
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
    src.append("    { let y = ").append(rhs).append(" - ").append(comp).append(";")
        .append(" let t = ").append(acc).append(" + y;")
        .append(' ').append(comp).append(" = (t - ").append(acc).append(") - y;")
        .append(' ').append(acc).append(" = t; }\n");
  }

  /**
   * One forward node for every market, declared ({@code var v<i> : f32}) or
   * assigned to an already-declared variable.
   */
  private static void emitNode(StringBuilder src, AadTape tape, Config cfg, int i, boolean declare) {
    int markets = cfg.markets();
    int a = tape.argA(i);
    int b = tape.argB(i);
    AadOp op = tape.op(i);
    if (marketFree(op)) {
      String name = "v" + i;
      if (declare) {
        src.append("    var ").append(name).append(" : f32 = ");
      } else {
        src.append("    ").append(name).append(" = ");
      }
      src.append(op == AadOp.RANDN ? "rng_normal(" + a + "u)" : literal(tape.constant(i)));
      src.append(";\n");
      return;
    }
    for (int m = 0; m < markets; m++) {
      String va = a >= 0 ? vn(tape, a, m, markets) : "";
      String vb = b >= 0 ? vn(tape, b, m, markets) : "";
      String name = vn(tape, i, m, markets);
      if (declare) {
        src.append("    var ").append(name).append(" : f32 = ");
      } else {
        src.append("    ").append(name).append(" = ");
      }
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
          src.append("    var ").append(dn(i, m, markets)).append(" : f32 = 0.0;\n");
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
          case ABS -> src.append("    ").append(da).append(" += select(").append(di).append(", -").append(di)
              .append(", ").append(va).append(" < 0.0);\n");
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
      src.append("    if ").append(taken).append(" { ").append(da).append(" += ").append(di).append("; }\n");
    }
    if (activeB) {
      src.append("    if (!").append(taken).append(") { ").append(db).append(" += ").append(di).append("; }\n");
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
   * pure function of {@code (path, k)}.
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
      src.append("    rng_init((pathLo ^ pc[").append(PC_SEED_LO).append("u]) ^ pc[").append(PC_ZERO).append("u], ")
          .append("(pathHi ^ pc[").append(PC_SEED_HI).append("u]) ^ pc[").append(PC_ZERO).append("u]);\n");
      for (int j = 0; j < cuts[s]; j++) {
        if (lastUse[j] >= cuts[s]) {
          int copies = marketFree(tape.op(j)) ? 1 : markets;
          for (int m = 0; m < copies; m++) {
            String name = vn(tape, j, m, markets);
            src.append("    var ").append(name).append(" : f32 = opq(").append(name).append(");\n");
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
      src.append("    var ").append(value ? vn(tape, i, m, markets) : dn(i, m, markets)).append(" : f32");
      if (init != null) src.append(" = ").append(init);
      src.append(";\n");
    }
  }

  /** A finite single-precision WGSL literal via an explicit {@code f32(...)} conversion. */
  private static String literal(double value) {
    float f = (float) value;
    if (Float.isNaN(f) || Float.isInfinite(f)) {
      throw new IllegalArgumentException("tape constant is not finite: " + value);
    }
    String text = Float.toString(f).replace('E', 'e');
    if (text.indexOf('.') < 0 && text.indexOf('e') < 0) {
      text = text + ".0";
    }
    return "f32(" + text + ")";
  }

  /**
   * Philox2x32-10 keyed on the scenario index and indexed by draw number,
   * byte-for-byte the generator emitted into the CUDA/Vulkan kernels. Taking
   * the draw index {@code k} as an argument (instead of an internal counter)
   * makes a draw a pure function of {@code (path, k)}, so the adjoint sweep
   * can recompute a checkpointed segment's randoms without saving RNG state.
   * WGSL has no {@code umulExtended}, so the 32x32→64 multiply is the classic
   * split-into-16-bit-halves software emulation (the same one GLSL falls back
   * to on hardware without a native instruction), returned as a small struct.
   */
  private static void appendRng(StringBuilder src) {
    src.append("""
        struct Wide { hi: u32, lo: u32 }

        fn umul_extended(x: u32, y: u32) -> Wide {
          let x0 = x & 0xffffu; let x1 = x >> 16u;
          let y0 = y & 0xffffu; let y1 = y >> 16u;
          let p0 = x0 * y0; let p1 = x0 * y1; let p2 = x1 * y0; let p3 = x1 * y1;
          let carry = ((p0 >> 16u) + (p1 & 0xffffu) + (p2 & 0xffffu)) >> 16u;
          let lsb = p0 + ((p1 + p2) << 16u);
          let msb = p3 + (p1 >> 16u) + (p2 >> 16u) + carry;
          return Wide(msb, lsb);
        }

        var<private> g_lo: u32;
        var<private> g_hi: u32;

        fn rng_init(keyed_lo: u32, keyed_hi: u32) {
          g_lo = keyed_lo;
          g_hi = keyed_hi;
        }

        fn rng_normal(k: u32) -> f32 {
          var c0 = g_lo;
          var c1 = g_hi ^ (k >> 1u);
          var key = 0x1BD11BDAu;
          for (var i = 0; i < 10; i = i + 1) {
            let w = umul_extended(0xD256D193u, c0);
            c0 = w.hi ^ key ^ c1;
            c1 = w.lo;
            key = key + 0x9E3779B9u;
          }
          let u1 = (f32(c0) + 0.5) * 2.3283064365386963e-10;
          let u2 = (f32(c1) + 0.5) * 2.3283064365386963e-10;
          let radius = sqrt(-2.0 * log(u1));
          let angle = 6.283185307179586 * u2;
          return select(radius * sin(angle), radius * cos(angle), (k & 1u) == 0u);
        }
        """);
  }
}
