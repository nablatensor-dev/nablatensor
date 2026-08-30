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

/**
 * A backend able to turn a recorded tape into something replayable.
 *
 * <p>Implementations are discovered with {@link java.util.ServiceLoader}, so a
 * backend becomes usable simply by being on the module path. An implementation
 * that depends on an optional runtime capability — a CUDA device, an incubating
 * JDK module — must keep that dependency out of its own class body and report
 * it through {@link #isAvailable()}, so that merely enumerating engines cannot
 * fail on a machine that lacks it.
 */
public interface AadEngine {

  /** Stable lower-case identifier used by {@code -Dnablatensor.engine}. */
  String name();

  /**
   * Higher wins when selecting automatically. CUDA is 100, SIMD 50, scalar 10,
   * so an accelerator is always preferred unless one is named explicitly.
   */
  int priority();

  /** Whether this engine can run here, right now. Must never throw. */
  boolean isAvailable();

  /** Whether this engine can honour these options at all. */
  boolean supports(AadOptions options);

  /** Short human-readable description of what it will run on. */
  String describe();

  AadExecutable compile(AadTape tape, AadOptions options);

  /**
   * Guard for backends that only implement a single stream of standard-normal
   * draws: throws (so automatic selection falls back to one that can) when the
   * tape uses {@code rec.randu()} or a named {@code rec.stream(...)}. A backend
   * that has grown native support for those simply does not call this.
   */
  static void requireBasicRandom(AadTape tape, String engine) {
    if (tape.hasExtendedRandom()) {
      throw new UnsupportedOperationException(
          "engine '" + engine + "' does not yet support rec.randu() or named random streams; "
              + "use cpu-jit or cpu for this tape");
    }
  }

  /**
   * Guard for backends that only implement a single output: throws (so automatic
   * selection falls back) when the tape was recorded with several
   * {@code rec.output(name, ...)} calls.
   */
  static void requireSingleOutput(AadTape tape, String engine) {
    if (tape.outputCount() > 1) {
      throw new UnsupportedOperationException(
          "engine '" + engine + "' does not yet support multi-output tapes (" + tape.outputNames()
              + "); use cpu-jit or cpu for this tape");
    }
  }
}
