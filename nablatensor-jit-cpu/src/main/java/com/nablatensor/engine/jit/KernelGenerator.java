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
package com.nablatensor.engine.jit;

import com.nablatensor.engine.AadOp;
import com.nablatensor.engine.AadTape;

import java.lang.classfile.ClassBuilder;
import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;

import static java.lang.constant.ConstantDescs.CD_Object;
import static java.lang.constant.ConstantDescs.CD_double;
import static java.lang.constant.ConstantDescs.CD_float;
import static java.lang.constant.ConstantDescs.CD_void;
import static java.lang.constant.ConstantDescs.INIT_NAME;
import static java.lang.constant.ConstantDescs.MTD_void;

/**
 * Emits a tape-specific kernel class with the Class-File API: the forward and
 * reverse sweeps become straight-line bytecode with node indices and constants
 * baked in — no interpreter dispatch, no {@code ops[i]}/{@code argA[i]} array
 * loads.
 *
 * <p>The sweeps are split into {@code fwd$k}/{@code rev$k} methods of ≤
 * {@code segNodes} nodes so each stays under C2's {@code -XX:HugeMethodLimit};
 * values that cross a segment boundary live in the shared {@code v}/{@code d}
 * arrays. Per-op emission goes through {@link Slots}, so the same formulas serve
 * both this flat shape and (later) a rolled-loop shape whose values live in JVM
 * locals.
 *
 * <p>Both precisions come from one walk: fp64 → {@link JitKernel}, fp32 →
 * {@link JitKernelF32} with {@code float} arithmetic (transcendentals widen to
 * {@code double} for {@code Math}).
 */
final class KernelGenerator {

  private static final ClassDesc CD_DBL_ARR = CD_double.arrayType();
  private static final ClassDesc CD_FLT_ARR = CD_float.arrayType();
  private static final ClassDesc CD_MATH = ClassDesc.of("java.lang.Math");
  private static final ClassDesc CD_KERNEL = ClassDesc.of("com.nablatensor.engine.jit.JitKernel");
  private static final ClassDesc CD_KERNEL_F32 = ClassDesc.of("com.nablatensor.engine.jit.JitKernelF32");
  private static final ClassDesc CD_SELF = ClassDesc.of("com.nablatensor.engine.jit.JitKernel_Gen");

  /** {@code -Dnablatensor.jit.exp=none}: emit identity for EXP (probe, not correct). */
  private static final boolean EXP_NONE = "none".equals(System.getProperty("nablatensor.jit.exp"));
  /** {@code -Dnablatensor.jit.randn=zero}: emit 0 for every draw (probe, not correct). */
  private static final boolean RANDN_ZERO = "zero".equals(System.getProperty("nablatensor.jit.randn"));

  private static final MethodTypeDesc MTD_D_D = MethodTypeDesc.of(CD_double, CD_double);
  private static final MethodTypeDesc MTD_DD_D = MethodTypeDesc.of(CD_double, CD_double, CD_double);
  private static final MethodTypeDesc MTD_F_F = MethodTypeDesc.of(CD_float, CD_float);
  private static final MethodTypeDesc MTD_FF_F = MethodTypeDesc.of(CD_float, CD_float, CD_float);

  private KernelGenerator() {
  }

  /**
   * @param nodeSlot  {@code v[]}/{@code d[]} slot for each node ({@code -1} = the
   *     rolled kernel never touches it); identity for the flat kernel
   * @param vLen      length of the {@code v}/{@code d} arrays the kernel needs
   */
  private record Plan(Tape tp, Loop loop, int[] nodeSlot, int vLen) {}

  // One entry per tape, holding the (at most four) plans for the
  // adjoints × roll combinations that tape has been asked for.
  private static final java.util.Map<AadTape, Plan[]> PLAN_CACHE =
      java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

  private static int planSlot(boolean adjoints, boolean roll) {
    return (adjoints ? 1 : 0) | (roll ? 2 : 0);
  }

  private static Plan plan(AadTape tape, boolean adjoints, boolean roll) {
    Plan[] slots = PLAN_CACHE.computeIfAbsent(tape, key -> new Plan[4]);
    int ix = planSlot(adjoints, roll);
    Plan cached = slots[ix];
    if (cached != null) {
      return cached;
    }
    Tape tp = flatten(tape);
    Loop lp = (roll && adjoints) ? detectLoop(tp) : null;
    int[] nodeSlot = new int[tp.n];
    int vLen;
    if (lp == null) {
      for (int i = 0; i < tp.n; i++) {
        nodeSlot[i] = i;
      }
      vLen = tp.n;
    } else {
      // the rolled kernel only reads/writes v[]/d[] for the prologue, the
      // epilogue, and the last iteration's carry-out nodes.
      java.util.Arrays.fill(nodeSlot, -1);
      int next = 0;
      for (int i = 0; i < lp.bodyStart; i++) {
        nodeSlot[i] = next++;
      }
      for (int rel : lp.carryRel) {
        int lastIter = lp.bodyStart + (lp.iters - 1) * lp.period + rel;
        if (nodeSlot[lastIter] < 0) {
          nodeSlot[lastIter] = next++;
        }
      }
      for (int i = lp.epilogueStart; i < tp.n; i++) {
        if (nodeSlot[i] < 0) {
          nodeSlot[i] = next++;
        }
      }
      vLen = next;
    }
    Plan p = new Plan(tp, lp, nodeSlot, vLen);
    slots[ix] = p;
    return p;
  }

  /** Length of the {@code v}/{@code d} arrays {@link JitReplay} must allocate. */
  static int vLen(AadTape tape, boolean adjoints, boolean roll) {
    return plan(tape, adjoints, roll).vLen;
  }

  /** {@code v[]}/{@code d[]} slot for {@code node} (identity for the flat kernel). */
  static int mapNode(AadTape tape, boolean adjoints, boolean roll, int node) {
    return plan(tape, adjoints, roll).nodeSlot[node];
  }

  static Object generate(AadTape tape, boolean adjoints, boolean roll, int segNodes, boolean f32) {
    byte[] bytes = emit(plan(tape, adjoints, roll), adjoints, Math.max(8, segNodes), f32);
    try {
      Class<?> cls = MethodHandles.lookup().defineHiddenClass(bytes, true).lookupClass();
      return cls.getConstructor().newInstance();
    } catch (Throwable failure) {
      throw new IllegalStateException("AAD kernel generation failed", failure);
    }
  }

  static int classFileSize(AadTape tape, boolean adjoints, boolean roll, int segNodes, boolean f32) {
    return emit(plan(tape, adjoints, roll), adjoints, Math.max(8, segNodes), f32).length;
  }

  /** "flat" or e.g. "rolled(period=6 x250, tape=2)" — for diagnostics / the write-up. */
  static String describeShape(AadTape tape, boolean adjoints, boolean roll) {
    Loop lp = plan(tape, adjoints, roll).loop;
    return lp == null ? "flat"
        : "rolled(period=" + lp.period + " x" + lp.iters + ", tape=" + lp.tapedRel.size() + ")";
  }

  // ------------------------------------------------------------------ tape ---

  private record Tape(int n, AadOp[] op, int[] a, int[] b, double[] k, boolean[] active,
                      int[] drawIx, int output) {}

  private static Tape flatten(AadTape tape) {
    int n = tape.size();
    AadOp[] op = new AadOp[n];
    int[] a = new int[n];
    int[] b = new int[n];
    double[] k = new double[n];
    boolean[] active = new boolean[n];
    int[] drawIx = new int[n];
    for (int i = 0; i < n; i++) {
      op[i] = tape.op(i);
      a[i] = tape.argA(i);
      b[i] = tape.argB(i);
      k[i] = tape.constant(i);
      active[i] = tape.isActive(i);
      drawIx[i] = (op[i] == AadOp.RANDN || op[i] == AadOp.RANDU) ? tape.randFlatIndex(i) : -1;
    }
    return new Tape(n, op, a, b, k, active, drawIx, tape.outputNode());
  }

  // ------------------------------------------------------------------ emit ---

  /** {@code double[]} step-tape elements the rolled kernel needs (0 = flat). */
  static int scratchLen(AadTape tape, boolean adjoints, boolean roll) {
    Loop lp = plan(tape, adjoints, roll).loop;
    if (lp == null) {
      return 0;
    }
    int total = 0;
    for (int rel : lp.tapedRel) {
      total += lp.carryRel.contains(rel) ? lp.iters + 1 : lp.iters;
    }
    return total;
  }

  private static byte[] emit(Plan p, boolean adjoints, int seg, boolean f32) {
    Tape tp = p.tp;
    Loop lp = p.loop;
    T t = new T(f32);
    return ClassFile.of().build(CD_SELF, clb -> {
      clb.withFlags(ClassFile.ACC_PUBLIC | ClassFile.ACC_FINAL);
      clb.withSuperclass(CD_Object);
      clb.withInterfaceSymbols(f32 ? CD_KERNEL_F32 : CD_KERNEL);
      clb.withMethodBody(INIT_NAME, MTD_void, ClassFile.ACC_PUBLIC, cb ->
          cb.aload(0).invokespecial(CD_Object, INIT_NAME, MTD_void).return_());
      if (lp != null) {
        emitRolled(clb, tp, lp, p.nodeSlot, t);
      } else {
        emitFlat(clb, tp, adjoints, seg, t);
      }
    });
  }

  // ============================================================ flat shape ===

  private static void emitFlat(ClassBuilder clb, Tape tp, boolean adjoints, int seg, T t) {
    int n = tp.n;
    List<int[]> fwdSegs = new ArrayList<>();
    for (int s = 0; s < n; s += seg) {
      fwdSegs.add(new int[] {s, Math.min(n, s + seg)});
    }
    List<List<Integer>> revSegs = new ArrayList<>();
    if (adjoints) {
      List<Integer> cur = new ArrayList<>();
      for (int i = n - 1; i >= 0; i--) {
        if (!revEmit(tp, i)) {
          continue;
        }
        cur.add(i);
        if (cur.size() == seg) {
          revSegs.add(cur);
          cur = new ArrayList<>();
        }
      }
      if (!cur.isEmpty()) {
        revSegs.add(cur);
      }
    }
    Slots fwdSlots = new ArraySlots(t, tp, null, 0, -1, 1, 2);   // fwd$k(v, in, draws)
    Slots revSlots = new ArraySlots(t, tp, null, 0, 1, -1, -1);  // rev$k(v, d)

    for (int si = 0; si < fwdSegs.size(); si++) {
      int[] r = fwdSegs.get(si);
      clb.withMethodBody("fwd$" + si, t.segType(), ClassFile.ACC_PRIVATE | ClassFile.ACC_STATIC, cb -> {
        for (int i = r[0]; i < r[1]; i++) {
          fwdNode(cb, tp, i, fwdSlots, t);
        }
        cb.return_();
      });
    }
    clb.withMethodBody("forward", t.fwdType(), ClassFile.ACC_PUBLIC, cb -> {   // (v,in,draws,scratch)
      for (int si = 0; si < fwdSegs.size(); si++) {
        cb.aload(1).aload(2).aload(3).invokestatic(CD_SELF, "fwd$" + si, t.segType());
      }
      cb.aload(1).loadConstant(tp.output).arrayLoad(t.tk());
      t.ret(cb);
    });

    for (int si = 0; si < revSegs.size(); si++) {
      List<Integer> nodes = revSegs.get(si);
      clb.withMethodBody("rev$" + si, t.revSegType(), ClassFile.ACC_PRIVATE | ClassFile.ACC_STATIC, cb -> {
        for (int i : nodes) {
          revNode(cb, tp, i, revSlots, t);
        }
        cb.return_();
      });
    }
    clb.withMethodBody("reverse", t.revType(), ClassFile.ACC_PUBLIC, cb -> {   // (v,d,scratch)
      for (int si = 0; si < revSegs.size(); si++) {
        cb.aload(1).aload(2).invokestatic(CD_SELF, "rev$" + si, t.revSegType());
      }
      cb.return_();
    });
  }

  private static boolean revEmit(Tape tp, int i) {
    return tp.active[i] && tp.op[i] != AadOp.CONST && tp.op[i] != AadOp.INPUT && tp.op[i] != AadOp.RANDN;
  }

  // ========================================================== rolled shape ===

  private static void emitRolled(ClassBuilder clb, Tape tp, Loop lp, int[] slot, T t) {
    final int period = lp.period, iters = lp.iters, bodyStart = lp.bodyStart;
    final int firstCopy = bodyStart - period;

    // step-tape layout: base per taped rel; carries get iters+1 (index 0 = P[-1]).
    final int[] tapeBase = new int[period];
    java.util.Arrays.fill(tapeBase, -1);
    int cur = 0;
    for (int rel : lp.tapedRel) {
      tapeBase[rel] = cur;
      cur += lp.carryRel.contains(rel) ? iters + 1 : iters;
    }

    // ---- forward(this0, v1, in2, draws3, scratch4) ----
    clb.withMethodBody("forward", t.fwdType(), ClassFile.ACC_PUBLIC, cb -> {
      final int itSlot = 5;
      int nxt = 6;
      final int drawIdxSlot = nxt++;                 // int: firstOrd + it*randPerIter
      java.util.Map<Integer, Integer> carrySlot = new java.util.HashMap<>();
      for (int rel : lp.carryRel) { carrySlot.put(rel, nxt); nxt += 2; }
      int[] blSlot = new int[period];
      for (int j = 0; j < period; j++) { blSlot[j] = nxt; nxt += 2; }
      java.util.Map<Integer, Integer> invLocal = new java.util.HashMap<>();
      for (int inv : lp.invNodes) { invLocal.put(inv, nxt); nxt += 2; }

      Slots prol = new ArraySlots(t, tp, slot, 1, -1, 2, 3);
      for (int i = 0; i < bodyStart; i++) {
        fwdNode(cb, tp, i, prol, t);
      }
      // hoist the invariants the body reads out of v[] into locals
      for (int inv : lp.invNodes) {
        cb.aload(1).loadConstant(slot[inv]).arrayLoad(t.tk());
        cb.storeLocal(t.tk(), invLocal.get(inv));
      }
      cb.loadConstant(lp.firstRandOrd).istore(drawIdxSlot);
      // seed carry locals; for a taped carry also seed step-tape slot 0 (= P[-1])
      for (int rel : lp.carryRel) {
        cb.aload(1).loadConstant(slot[firstCopy + rel]).arrayLoad(t.tk());
        cb.storeLocal(t.tk(), carrySlot.get(rel));
        if (tapeBase[rel] >= 0) {
          cb.aload(4).loadConstant(tapeBase[rel]);
          cb.aload(1).loadConstant(slot[firstCopy + rel]).arrayLoad(t.tk());
          cb.arrayStore(t.tk());
        }
      }
      for (int j = 0; j < period; j++) { t.load(cb, 0.0); cb.storeLocal(t.tk(), blSlot[j]); }

      cb.loadConstant(0).istore(itSlot);
      Label top = cb.newLabel();
      Label end = cb.newLabel();
      cb.labelBinding(top);
      cb.iload(itSlot).loadConstant(iters).if_icmpge(end);
      RolledFwdSlots body = new RolledFwdSlots(t, tp, lp, blSlot, carrySlot, invLocal, itSlot, drawIdxSlot, tapeBase, firstCopy);
      for (int j = 0; j < period; j++) {
        fwdNode(cb, tp, bodyStart + j, body, t);
      }
      for (int rel : lp.carryRel) {
        cb.loadLocal(t.tk(), blSlot[rel]).storeLocal(t.tk(), carrySlot.get(rel));
      }
      if (lp.randPerIter > 0) {
        cb.iinc(drawIdxSlot, lp.randPerIter);
      }
      cb.iinc(itSlot, 1).goto_(top);
      cb.labelBinding(end);
      // hand the final carry-outs to the flat epilogue via v[]
      for (int rel : lp.carryRel) {
        cb.aload(1).loadConstant(slot[bodyStart + (iters - 1) * period + rel]);
        cb.loadLocal(t.tk(), carrySlot.get(rel)).arrayStore(t.tk());
      }
      for (int i = lp.epilogueStart; i < tp.n; i++) {
        fwdNode(cb, tp, i, prol, t);
      }
      cb.aload(1).loadConstant(slot[tp.output]).arrayLoad(t.tk());
      t.ret(cb);
    });

    // ---- reverse(this0, v1, d2, scratch3, draws4) ----
    clb.withMethodBody("reverse", t.revType(), ClassFile.ACC_PUBLIC, cb -> {
      final int itSlot = 5;
      int nxt = 6;
      java.util.Map<Integer, Integer> barCarry = new java.util.HashMap<>();
      for (int rel : lp.carryRel) { barCarry.put(rel, nxt); nxt += 2; }
      int[] barBl = new int[period];
      java.util.Arrays.fill(barBl, -1);
      for (int j = 0; j < period; j++) {
        if (revEmit(tp, bodyStart + j)) { barBl[j] = nxt; nxt += 2; }
      }

      Slots flat = new ArraySlots(t, tp, slot, 1, 2, -1, -1);
      for (int i = tp.n - 1; i >= lp.epilogueStart; i--) {
        if (revEmit(tp, i)) revNode(cb, tp, i, flat, t);
      }
      for (int rel : lp.carryRel) {
        cb.aload(2).loadConstant(slot[bodyStart + (iters - 1) * period + rel]).arrayLoad(t.tk());
        cb.storeLocal(t.tk(), barCarry.get(rel));
      }
      for (int j = 0; j < period; j++) {
        if (barBl[j] >= 0) { t.load(cb, 0.0); cb.storeLocal(t.tk(), barBl[j]); }
      }

      cb.loadConstant(iters - 1).istore(itSlot);
      Label top = cb.newLabel();
      Label end = cb.newLabel();
      cb.labelBinding(top);
      cb.iload(itSlot).loadConstant(0).if_icmplt(end);
      // per-iteration init: carry-out adjoints take the incoming barCarry, then barCarry is zeroed
      for (int j = 0; j < period; j++) {
        if (barBl[j] < 0) continue;
        if (lp.carryRel.contains(j)) {
          cb.loadLocal(t.tk(), barCarry.get(j)).storeLocal(t.tk(), barBl[j]);
          t.load(cb, 0.0);
          cb.storeLocal(t.tk(), barCarry.get(j));
        } else {
          t.load(cb, 0.0);
          cb.storeLocal(t.tk(), barBl[j]);
        }
      }
      RolledRevSlots rb = new RolledRevSlots(t, tp, lp, barBl, barCarry, itSlot, tapeBase, firstCopy);
      for (int j = period - 1; j >= 0; j--) {
        if (revEmit(tp, bodyStart + j)) revNode(cb, tp, bodyStart + j, rb, t);
      }
      cb.iinc(itSlot, -1).goto_(top);
      cb.labelBinding(end);

      for (int rel : lp.carryRel) {
        cb.aload(2).loadConstant(slot[firstCopy + rel]).dup2().arrayLoad(t.tk());
        cb.loadLocal(t.tk(), barCarry.get(rel));
        t.add(cb);
        cb.arrayStore(t.tk());
      }
      for (int i = bodyStart - 1; i >= 0; i--) {
        if (revEmit(tp, i)) revNode(cb, tp, i, flat, t);
      }
      cb.return_();
    });
  }

  /** Forward body: values in {@code blSlot} locals, carries in {@code carrySlot}, taped to {@code scratch}. */
  private record RolledFwdSlots(T t, Tape tp, Loop lp, int[] blSlot, java.util.Map<Integer, Integer> carrySlot,
                                java.util.Map<Integer, Integer> invLocal, int itSlot, int drawIdxSlot,
                                int[] tapeBase, int firstCopy) implements Slots {
    public void storeV(CodeBuilder cb, int node, Runnable value) {
      int j = node - lp.bodyStart();
      value.run();
      cb.storeLocal(t.tk(), blSlot[j]);
      if (tapeBase[j] >= 0) {
        int off = lp.carryRel().contains(j) ? 1 : 0;
        cb.aload(4).loadConstant(tapeBase[j] + off).iload(itSlot).iadd();
        cb.loadLocal(t.tk(), blSlot[j]);
        cb.arrayStore(t.tk());
      }
    }
    public void loadV(CodeBuilder cb, int node) {
      int bs = lp.bodyStart();
      if (node >= bs && node < bs + lp.period()) {
        cb.loadLocal(t.tk(), blSlot[node - bs]);
      } else if (node >= firstCopy && node < bs) {
        cb.loadLocal(t.tk(), carrySlot.get(node - firstCopy));
      } else {
        cb.loadLocal(t.tk(), invLocal.get(node));      // hoisted invariant
      }
    }
    public void loadDraw(CodeBuilder cb, int node) {
      int rank = 0;
      for (int q = lp.bodyStart(); q < node; q++) {
        if (tp.op()[q] == AadOp.RANDN) rank++;
      }
      cb.aload(3).iload(drawIdxSlot);
      if (rank != 0) {
        cb.loadConstant(rank).iadd();
      }
      cb.arrayLoad(t.tk());
    }
    public void loadInput(CodeBuilder cb, int idx) { throw new IllegalStateException("INPUT in rolled body"); }
    public void loadD(CodeBuilder cb, int node) { throw new IllegalStateException(); }
    public void addD(CodeBuilder cb, int node, Runnable delta) { throw new IllegalStateException(); }
  }

  /** Reverse body: adjoints in {@code barBl}, carry adjoints in {@code barCarry}, invariant reductions in {@code barInv}. */
  private record RolledRevSlots(T t, Tape tp, Loop lp, int[] barBl, java.util.Map<Integer, Integer> barCarry,
                                int itSlot, int[] tapeBase, int firstCopy) implements Slots {
    public void loadV(CodeBuilder cb, int node) {
      int bs = lp.bodyStart();
      if (node >= bs && node < bs + lp.period()) {                 // this iteration's body value
        int j = node - bs;
        if (tp.op()[node] == AadOp.RANDN) {                       // a draw — read it back directly
          int rank = 0;
          for (int q = bs; q < node; q++) {
            if (tp.op()[q] == AadOp.RANDN) rank++;
          }
          cb.aload(4).loadConstant(lp.firstRandOrd() + rank);
          if (lp.randPerIter() == 1) {
            cb.iload(itSlot).iadd();
          } else {
            cb.iload(itSlot).loadConstant(lp.randPerIter()).imul().iadd();
          }
          cb.arrayLoad(t.tk());
          return;
        }
        int off = lp.carryRel().contains(j) ? 1 : 0;              // taped body value
        cb.aload(3).loadConstant(tapeBase[j] + off).iload(itSlot).iadd();
        cb.arrayLoad(t.tk());
      } else if (node >= firstCopy && node < bs) {                 // carry-in P[it-1]
        int rel = node - firstCopy;
        cb.aload(3).loadConstant(tapeBase[rel]).iload(itSlot).iadd();
        cb.arrayLoad(t.tk());
      } else {                                                     // invariant
        cb.aload(1).loadConstant(node).arrayLoad(t.tk());
      }
    }
    public void loadD(CodeBuilder cb, int node) {
      cb.loadLocal(t.tk(), barBl[node - lp.bodyStart()]);
    }
    public void addD(CodeBuilder cb, int node, Runnable delta) {
      int bs = lp.bodyStart();
      if (node >= bs && node < bs + lp.period()) {
        int slot = barBl[node - bs];
        if (slot < 0) { delta.run(); t.pop(cb); return; }          // inactive body node — discard
        cb.loadLocal(t.tk(), slot);
        delta.run();
        t.add(cb);
        cb.storeLocal(t.tk(), slot);
      } else if (node >= firstCopy && node < bs) {
        int slot = barCarry.get(node - firstCopy);
        cb.loadLocal(t.tk(), slot);
        delta.run();
        t.add(cb);
        cb.storeLocal(t.tk(), slot);
      } else {
        // invariant: accumulate straight into d[node] each iteration, matching
        // the flat kernel's summation order (so the result stays bit-exact).
        cb.aload(2).loadConstant(node).dup2().arrayLoad(t.tk());
        delta.run();
        t.add(cb);
        cb.arrayStore(t.tk());
      }
    }
    public void storeV(CodeBuilder cb, int node, Runnable value) { throw new IllegalStateException(); }
    public void loadInput(CodeBuilder cb, int idx) { throw new IllegalStateException(); }
    public void loadDraw(CodeBuilder cb, int node) { throw new IllegalStateException(); }
  }

  // ============================================================ loop detect ===

  /**
   * A detected simulation loop: prologue {@code [0, bodyStart)} · body ×
   * {@code iters} · epilogue {@code [epilogueStart, n)}. The block just before
   * {@code bodyStart} ({@code [bodyStart-period, bodyStart)}) is the loop's
   * first copy — it stays in the prologue and seeds the carry variables.
   */
  record Loop(int bodyStart, int period, int iters, int epilogueStart,
              int firstRandOrd, int randPerIter,
              List<Integer> carryRel,    // body-rel positions carried to the next iteration
              List<Integer> tapedRel,    // body-rel positions the reverse reads per iteration
              List<Integer> invNodes) {  // pre-body nodes the body reads (adjoint accumulates)
  }

  private static Loop detectLoop(Tape tp) {
    int n = tp.n;
    int bestStart = -1, bestPeriod = 0, bestIters = 0;
    for (int period = 3; period <= 40; period++) {
      for (int start = period; start + 2 * period <= n; start++) {
        int iters = uniformRepeats(tp, start, period);
        if (iters >= 8 && iters * period > bestIters * bestPeriod) {
          bestStart = start;
          bestPeriod = period;
          bestIters = iters;
        }
      }
    }
    if (bestStart < 0) {
      return null;
    }
    final int start = bestStart, period = bestPeriod, iters = bestIters;
    final int epi = start + iters * period;

    // body ops must all be localisable. CONST/INPUT never appear in a
    // recorder-emitted loop body; bail if they do (a per-iteration CONST would
    // also need a k[] equality check that uniformRepeats does not do).
    for (int j = 0; j < period; j++) {
      switch (tp.op[start + j]) {
        case ADD, SUB, MUL, DIV, NEG, EXP, LOG, SQRT, ABS, MAX, MIN, RANDN -> { }
        default -> { return null; }
      }
    }

    int randPerIter = 0;
    int firstOrd = 0;
    for (int j = 0; j < period; j++) {
      if (tp.op[start + j] == AadOp.RANDN) {
        if (randPerIter == 0) {
          firstOrd = tp.a[start + j];
        }
        randPerIter++;
      }
    }

    // carry-outs: block-0 nodes referenced by block 1
    boolean[] carry = new boolean[period];
    for (int j = 0; j < period; j++) {
      int q = start + period + j;
      for (int arg : argRefs(tp, q)) {
        if (arg >= start && arg < start + period) {
          carry[arg - start] = true;
        }
      }
    }
    List<Integer> carryRel = new ArrayList<>();
    for (int j = 0; j < period; j++) {
      if (carry[j]) {
        if (tp.op[start + j] == AadOp.RANDN) {
          return null;   // a draw carried to the next iteration (noise-MA) — not modelled; bail
        }
        carryRel.add(j);
      }
    }
    if (carryRel.isEmpty()) {
      return null;
    }

    // invariant pre-body nodes the body reads; body-rel nodes the reverse needs
    // as a step-tape (RANDN excluded — the reverse gets the draws array).
    List<Integer> invNodes = new ArrayList<>();
    boolean[] invSeen = new boolean[n];
    boolean[] tapedMask = new boolean[period];
    for (int j = 0; j < period; j++) {
      int i = start + j;
      for (int arg : argRefs(tp, i)) {
        if (arg < start - period) {
          if (!invSeen[arg]) {
            invSeen[arg] = true;
            invNodes.add(arg);
          }
        }
      }
      for (int side : revReads(tp.op[i])) {
        int arg = side == -1 ? i : (side == 0 ? tp.a[i] : tp.b[i]);
        if (arg >= start && arg < start + period) {
          if (tp.op[arg] != AadOp.RANDN) {
            tapedMask[arg - start] = true;      // per-iteration body value
          }
        } else if (arg >= start - period && arg < start && carry[arg - (start - period)]) {
          tapedMask[arg - (start - period)] = true;   // this carry's per-iteration value
        }
      }
    }
    List<Integer> tapedRel = new ArrayList<>();
    for (int j = 0; j < period; j++) {
      if (tapedMask[j]) {
        tapedRel.add(j);
      }
    }

    // epilogue may only read invariants or the final iteration's carry-outs
    for (int i = epi; i < n; i++) {
      for (int arg : argRefs(tp, i)) {
        if (arg >= start && arg < epi) {
          int rel = (arg - start) % period;
          int iter = (arg - start) / period;
          if (iter != iters - 1 || !carry[rel]) {
            return null;
          }
        }
      }
    }
    return new Loop(start, period, iters, epi, firstOrd, randPerIter, carryRel, tapedRel, invNodes);
  }

  /** From block {@code [start, start+period)}, how many consecutive blocks repeat. */
  private static int uniformRepeats(Tape tp, int start, int period) {
    int rpi = 0;
    for (int j = 0; j < period; j++) {
      if (tp.op[start + j] == AadOp.RANDN) {
        rpi++;
      }
    }
    int m = 1;
    while (start + (m + 1) * period <= tp.n) {
      boolean ok = true;
      for (int j = 0; j < period && ok; j++) {
        int r = start + (m - 1) * period + j;
        int q = start + m * period + j;
        if (tp.op[q] != tp.op[r]) {
          ok = false;
          break;
        }
        if (tp.op[q] == AadOp.RANDN) {
          if (tp.a[q] != tp.a[r] + rpi) {
            ok = false;
          }
          continue;
        }
        int[] rArgs = argRefs(tp, r);
        int[] qArgs = argRefs(tp, q);
        for (int x = 0; x < rArgs.length; x++) {
          int ar = rArgs[x];
          int aq = qArgs[x];
          if (ar >= start - period) {          // iteration-relative
            if (aq != ar + period) {
              ok = false;
              break;
            }
          } else {                             // loop-invariant
            if (aq != ar) {
              ok = false;
              break;
            }
          }
        }
      }
      if (!ok) {
        break;
      }
      m++;
    }
    return m;
  }

  private static int[] argRefs(Tape tp, int i) {
    return switch (tp.op[i]) {
      case CONST, INPUT, RANDN, RANDU -> new int[0];
      case NEG, EXP, LOG, SQRT, ABS -> new int[] {tp.a[i]};
      default -> new int[] {tp.a[i], tp.b[i]};
    };
  }

  /** Which {@code v[.]} the reverse rule for {@code op} reads: -1=this node, 0=a, 1=b. */
  private static int[] revReads(AadOp op) {
    return switch (op) {
      case MUL, MAX, MIN -> new int[] {0, 1};
      case DIV -> new int[] {1, -1};
      case EXP, SQRT -> new int[] {-1};
      case LOG, ABS -> new int[] {0};
      default -> new int[0];
    };
  }

  // ================================================= shared per-op emitters ===

  /** Where a node's value / adjoint / an input / a draw is read and written. */
  interface Slots {
    /** {@code v[node] = value.run()} — {@code value} pushes exactly one number. */
    void storeV(CodeBuilder cb, int node, Runnable value);
    /** push {@code v[node]}. */
    void loadV(CodeBuilder cb, int node);
    /** push {@code in[inputIdx]} (forward only). */
    void loadInput(CodeBuilder cb, int inputIdx);
    /** push the draw for RANDN node {@code node} (forward only). */
    void loadDraw(CodeBuilder cb, int node);
    /** push {@code d[node]}. */
    void loadD(CodeBuilder cb, int node);
    /** {@code d[node] += delta.run()} — {@code delta} pushes exactly one number. */
    void addD(CodeBuilder cb, int node, Runnable delta);
  }

  private static void fwdNode(CodeBuilder cb, Tape tp, int i, Slots s, T t) {
    int a = tp.a[i], b = tp.b[i];
    switch (tp.op[i]) {
      case CONST -> s.storeV(cb, i, () -> t.load(cb, tp.k[i]));
      case INPUT -> s.storeV(cb, i, () -> s.loadInput(cb, a));
      case RANDN, RANDU -> s.storeV(cb, i, () -> { if (RANDN_ZERO) t.load(cb, 0.0); else s.loadDraw(cb, i); });
      case ADD -> s.storeV(cb, i, () -> { s.loadV(cb, a); s.loadV(cb, b); t.add(cb); });
      case SUB -> s.storeV(cb, i, () -> { s.loadV(cb, a); s.loadV(cb, b); t.sub(cb); });
      case MUL -> s.storeV(cb, i, () -> { s.loadV(cb, a); s.loadV(cb, b); t.mul(cb); });
      case DIV -> s.storeV(cb, i, () -> { s.loadV(cb, a); s.loadV(cb, b); t.div(cb); });
      case NEG -> s.storeV(cb, i, () -> { s.loadV(cb, a); t.neg(cb); });
      case EXP -> s.storeV(cb, i, () -> { s.loadV(cb, a); if (!EXP_NONE) t.math1(cb, "exp"); });
      case LOG -> s.storeV(cb, i, () -> { s.loadV(cb, a); t.math1(cb, "log"); });
      case SQRT -> s.storeV(cb, i, () -> { s.loadV(cb, a); t.math1(cb, "sqrt"); });
      case ABS -> s.storeV(cb, i, () -> { s.loadV(cb, a); cb.invokestatic(CD_MATH, "abs", t.f32() ? MTD_F_F : MTD_D_D); });
      case MAX -> s.storeV(cb, i, () -> { s.loadV(cb, a); s.loadV(cb, b); cb.invokestatic(CD_MATH, "max", t.mathBin()); });
      case MIN -> s.storeV(cb, i, () -> { s.loadV(cb, a); s.loadV(cb, b); cb.invokestatic(CD_MATH, "min", t.mathBin()); });
    }
  }

  private static void revNode(CodeBuilder cb, Tape tp, int i, Slots s, T t) {
    int a = tp.a[i], b = tp.b[i];
    switch (tp.op[i]) {
      case ADD -> {
        s.addD(cb, a, () -> s.loadD(cb, i));
        s.addD(cb, b, () -> s.loadD(cb, i));
      }
      case SUB -> {
        s.addD(cb, a, () -> s.loadD(cb, i));
        s.addD(cb, b, () -> { s.loadD(cb, i); t.neg(cb); });
      }
      case NEG -> s.addD(cb, a, () -> { s.loadD(cb, i); t.neg(cb); });
      case MUL -> {
        s.addD(cb, a, () -> { s.loadD(cb, i); s.loadV(cb, b); t.mul(cb); });
        s.addD(cb, b, () -> { s.loadD(cb, i); s.loadV(cb, a); t.mul(cb); });
      }
      case DIV -> {
        s.addD(cb, a, () -> { s.loadD(cb, i); s.loadV(cb, b); t.div(cb); });
        s.addD(cb, b, () -> { s.loadD(cb, i); s.loadV(cb, i); t.mul(cb); s.loadV(cb, b); t.div(cb); t.neg(cb); });
      }
      case EXP -> s.addD(cb, a, () -> { s.loadD(cb, i); s.loadV(cb, i); t.mul(cb); });
      case LOG -> s.addD(cb, a, () -> { s.loadD(cb, i); s.loadV(cb, a); t.div(cb); });
      case SQRT -> s.addD(cb, a, () -> { s.loadD(cb, i); t.load(cb, 0.5); t.mul(cb); s.loadV(cb, i); t.div(cb); });
      case ABS -> s.addD(cb, a, () -> {
        t.load(cb, 1.0);
        s.loadV(cb, a);
        cb.invokestatic(CD_MATH, "copySign", t.mathBin());
        s.loadD(cb, i);
        t.mul(cb);
      });
      case MAX -> {
        s.loadV(cb, a);
        s.loadV(cb, b);
        t.cmpg(cb);
        cb.ifThenElse(Opcode.IFGE,
            x -> s.addD(x, a, () -> s.loadD(x, i)),
            x -> s.addD(x, b, () -> s.loadD(x, i)));
      }
      case MIN -> {
        s.loadV(cb, a);
        s.loadV(cb, b);
        t.cmpg(cb);
        cb.ifThenElse(Opcode.IFLE,
            x -> s.addD(x, a, () -> s.loadD(x, i)),
            x -> s.addD(x, b, () -> s.loadD(x, i)));
      }
      // ops that carry no adjoint back to their inputs
      case CONST, INPUT, RANDN, RANDU -> { }
      default -> throw new IllegalStateException("revNode: no reverse rule for op " + tp.op[i]);
    }
  }

  // ------------------------------------------------------------- slot impls ---

  /** Node value/adjoint live in {@code v[]}/{@code d[]} arrays; slot {@code -1} = absent. */
  /** {@code remap} compacts sparse node indices for the rolled kernel; {@code null} = identity. */
  private record ArraySlots(T t, Tape tp, int[] remap, int vSlot, int dSlot, int inSlot, int drawSlot)
      implements Slots {
    private int s(int node) { return remap == null ? node : remap[node]; }
    public void storeV(CodeBuilder cb, int node, Runnable value) {
      cb.aload(vSlot).loadConstant(s(node));
      value.run();
      cb.arrayStore(t.tk());
    }
    public void loadV(CodeBuilder cb, int node) {
      cb.aload(vSlot).loadConstant(s(node)).arrayLoad(t.tk());
    }
    public void loadInput(CodeBuilder cb, int inputIdx) {
      cb.aload(inSlot).loadConstant(inputIdx).arrayLoad(t.tk());
    }
    public void loadDraw(CodeBuilder cb, int node) {
      cb.aload(drawSlot).loadConstant(tp.drawIx[node]).arrayLoad(t.tk());
    }
    public void loadD(CodeBuilder cb, int node) {
      cb.aload(dSlot).loadConstant(s(node)).arrayLoad(t.tk());
    }
    public void addD(CodeBuilder cb, int node, Runnable delta) {
      cb.aload(dSlot).loadConstant(s(node)).dup2().arrayLoad(t.tk());
      delta.run();
      t.add(cb);
      cb.arrayStore(t.tk());
    }
  }

  // ----------------------------------------------------------- type helpers ---

  private record T(boolean f32) {
    TypeKind tk() { return f32 ? TypeKind.FLOAT : TypeKind.DOUBLE; }
    ClassDesc arr() { return f32 ? CD_FLT_ARR : CD_DBL_ARR; }
    ClassDesc prim() { return f32 ? CD_float : CD_double; }
    MethodTypeDesc segType() { return MethodTypeDesc.of(CD_void, arr(), arr(), arr()); }        // fwd$k(v,in,draws)
    MethodTypeDesc revSegType() { return MethodTypeDesc.of(CD_void, arr(), arr()); }            // rev$k(v,d)
    MethodTypeDesc fwdType() { return MethodTypeDesc.of(prim(), arr(), arr(), arr(), arr()); }  // forward(v,in,draws,scratch)
    MethodTypeDesc revType() { return MethodTypeDesc.of(CD_void, arr(), arr(), arr(), arr()); } // reverse(v,d,scratch,draws)
    MethodTypeDesc mathBin() { return f32 ? MTD_FF_F : MTD_DD_D; }

    void load(CodeBuilder cb, double c) { if (f32) cb.loadConstant((float) c); else cb.loadConstant(c); }
    void add(CodeBuilder cb) { if (f32) cb.fadd(); else cb.dadd(); }
    void sub(CodeBuilder cb) { if (f32) cb.fsub(); else cb.dsub(); }
    void mul(CodeBuilder cb) { if (f32) cb.fmul(); else cb.dmul(); }
    void div(CodeBuilder cb) { if (f32) cb.fdiv(); else cb.ddiv(); }
    void neg(CodeBuilder cb) { if (f32) cb.fneg(); else cb.dneg(); }
    void cmpg(CodeBuilder cb) { if (f32) cb.fcmpg(); else cb.dcmpg(); }
    void ret(CodeBuilder cb) { if (f32) cb.freturn(); else cb.dreturn(); }
    void pop(CodeBuilder cb) { if (f32) cb.pop(); else cb.pop2(); }

    void math1(CodeBuilder cb, String name) {
      if (f32) {
        cb.f2d().invokestatic(CD_MATH, name, MTD_D_D).d2f();
      } else {
        cb.invokestatic(CD_MATH, name, MTD_D_D);
      }
    }
  }
}
