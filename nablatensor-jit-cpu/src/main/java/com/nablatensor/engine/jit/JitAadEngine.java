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

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadExecutable;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadTape;

/**
 * Bytecode-generating replay engine. Priority 7 — below the scalar {@code cpu}
 * engine, so {@code fastest()} never picks it; request it by name ({@code
 * -Dnablatensor.engine=cpu-jit} / {@code .on("cpu-jit")}). fp64 and fp32: the
 * generated kernel's arithmetic is in the
 * requested precision, with per-scenario totals always accumulated in double.
 *
 * <p>Needs the Class-File API, final since JDK 24, so it runs unflagged on
 * Java 25 LTS; on an older JDK it reports itself unavailable.
 */
public final class JitAadEngine implements AadEngine {

  private static final boolean CLASSFILE_API =
      probe("java.lang.classfile.ClassFile");

  private static boolean probe(String className) {
    try {
      Class.forName(className, false, JitAadEngine.class.getClassLoader());
      return true;
    } catch (Throwable absent) {
      return false;
    }
  }

  @Override
  public String name() {
    return "cpu-jit";
  }

  @Override
  public int priority() {
    return 7;
  }

  @Override
  public boolean isAvailable() {
    return CLASSFILE_API;
  }

  @Override
  public boolean supports(AadOptions options) {
    return true;
  }

  @Override
  public String describe() {
    if (!CLASSFILE_API) {
      return "java.lang.classfile absent (needs JDK 24+)";
    }
    return "generated straight-line bytecode kernel · RNG prefilled · segmented for C2 · fp32+fp64";
  }

  @Override
  public AadExecutable compile(AadTape tape, AadOptions options) {
    if (!CLASSFILE_API) {
      throw new IllegalStateException("the cpu-jit engine needs the Class-File API (JDK 24+)");
    }
    return new JitReplay(tape, options);
  }
}
