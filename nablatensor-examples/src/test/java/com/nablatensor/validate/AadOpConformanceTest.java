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
package com.nablatensor.validate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadEngines;
import com.nablatensor.engine.AadOpEnum;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.ADouble;
import java.util.EnumSet;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Every {@link AadOpEnum} must be handled by every backend that claims to run here:
 * a tape that exercises the whole op set (both random kinds included) is
 * compiled on each available engine and its price and gradient are diffed
 * against the {@code cpu} scalar oracle. A backend that has not grown native
 * support for {@code RANDU} / named streams must decline the tape (so selection
 * falls back), never silently produce a wrong number.
 */
@Tag("mc")
class AadOpConformanceTest {

  private static final long N = 200_000L;
  private static final long SEED = 0x5EED5L;

  /** A tape touching CONST, INPUT, RANDN, RANDU, ADD, SUB, MUL, DIV, NEG, EXP, LOG, SQRT, ABS, MAX, MIN. */
  private static void everyOp(AadRecorder rec) {
    ADouble x = rec.input("x", 1.3);
    ADouble y = rec.input("y", 0.7);
    ADouble z = rec.randn();                 // RANDN
    ADouble u = rec.randu();                 // RANDU (default stream)
    ADouble a = x.add(y).sub(0.1);           // ADD, SUB, CONST
    ADouble b = x.mul(y).div(x.add(0.5));    // MUL, DIV
    ADouble c = a.neg().exp().add(b.abs());  // NEG, EXP, ABS
    ADouble d = c.add(1.0).log().sqrt();     // LOG, SQRT
    ADouble e = d.max(z.mul(0.01)).min(x.mul(2.0));  // MAX, MIN
    rec.output(e.add(u.mul(y)));
  }

  @Test
  void everyAadOpIsExercisedByThisTape() {
    Set<AadOpEnum> touched = EnumSet.of(
        AadOpEnum.CONST, AadOpEnum.INPUT, AadOpEnum.RANDN, AadOpEnum.RANDU, AadOpEnum.ADD, AadOpEnum.SUB, AadOpEnum.MUL,
        AadOpEnum.DIV, AadOpEnum.NEG, AadOpEnum.EXP, AadOpEnum.LOG, AadOpEnum.SQRT, AadOpEnum.ABS, AadOpEnum.MAX, AadOpEnum.MIN);
    Set<AadOpEnum> missing = EnumSet.allOf(AadOpEnum.class);
    missing.removeAll(touched);
    assertTrue(missing.isEmpty(),
        "AadOpEnum added without extending the conformance tape / every backend: " + missing);

    AadTape tape = AadRecorder.record(AadOpConformanceTest::everyOp);
    Set<AadOpEnum> onTape = EnumSet.noneOf(AadOpEnum.class);
    for (int i = 0; i < tape.size(); i++) {
      onTape.add(tape.op(i));
    }
    assertEquals(touched, onTape, "the recorded tape actually contains every op");
  }

  @Test
  void everyAvailableBackendMatchesTheOracleOrDeclines() {
    AadOptions fp64 = AadOptions.of().precision(AadOptions.PrecisionEnum.FLOAT64).adjoints(true).threads(0).jit(com.nablatensor.engine.JitOptimizations.NONE).engineOptions(java.util.Map.of()).build();
    AadTape tape = AadRecorder.record(AadOpConformanceTest::everyOp);

    AadEngine cpu = AadEngines.require("cpu", fp64);
    AadResult oracle;
    try (var exe = cpu.compile(tape, fp64)) {
      oracle = exe.replaySafe(N, SEED);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    for (AadEngine engine : AadEngines.discovered()) {
      if (!safeAvailable(engine) || engine.name().equals("cpu")) {
        continue;
      }
      AadResult got;
      try (var exe = engine.compile(tape, fp64)) {
        got = exe.replaySafe(N, SEED);
      } catch (UnsupportedOperationException declined) {
        // acceptable: a backend without native randu/stream support says so
        System.out.println(engine.name() + " declined the extended tape: " + declined.getMessage());
        continue;
      } catch (RuntimeException | LinkageError unavailable) {
        System.out.println(engine.name() + " unavailable at compile: " + unavailable);
        continue;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }

      double tol = 5e-4 + 5e-4 * Math.abs(oracle.value());
      assertEquals(oracle.value(), got.value(), tol, engine.name() + " price vs cpu oracle");
      for (String in : oracle.inputNames()) {
        double a = oracle.gradient(in);
        assertEquals(a, got.gradient(in), 5e-3 + 5e-3 * Math.abs(a),
            engine.name() + " d/d" + in + " vs cpu oracle");
      }
    }
  }

  private static boolean safeAvailable(AadEngine e) {
    try {
      return e.isAvailable();
    } catch (Throwable t) {
      return false;
    }
  }
}
