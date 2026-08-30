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
package com.nablatensor.scenario;

import java.util.function.DoubleUnaryOperator;

/**
 * A single named-input perturbation. Declarative data: the driver turns a set of
 * these into {@code setInput} calls on a compiled kernel (Seam 6).
 *
 * <p>{@link Kind#ABSOLUTE}, {@link Kind#RELATIVE} and {@link Kind#ADDITIVE} are
 * the common closed forms and ignore {@code transform}. {@link Kind#CUSTOM}
 * carries an arbitrary {@code base -> shocked} function for anything that does
 * not fit — a log-shock, a floor, a curve twist.
 *
 * @param input     the recorded input name to move
 * @param kind      how {@code amount} / {@code transform} is interpreted
 * @param amount    the shock size (unused for {@link Kind#CUSTOM})
 * @param transform the {@code base -> shocked} function for {@link Kind#CUSTOM}; {@code null} otherwise
 */
public record Shock(String input, Kind kind, double amount, DoubleUnaryOperator transform) {

  public enum Kind {
    /** Replace the input with {@code amount}. */
    ABSOLUTE,
    /** Multiply the base value by {@code (1 + amount)}. */
    RELATIVE,
    /** Add {@code amount} to the base value. */
    ADDITIVE,
    /** Apply {@code transform} to the base value. */
    CUSTOM
  }

  public Shock(String input, Kind kind, double amount) {
    this(input, kind, amount, null);
  }

  public Shock {
    if (kind == Kind.CUSTOM && transform == null) {
      throw new IllegalArgumentException("a CUSTOM shock needs a transform function");
    }
  }

  public static Shock absolute(String input, double value) {
    return new Shock(input, Kind.ABSOLUTE, value);
  }

  public static Shock relative(String input, double fraction) {
    return new Shock(input, Kind.RELATIVE, fraction);
  }

  public static Shock additive(String input, double delta) {
    return new Shock(input, Kind.ADDITIVE, delta);
  }

  /** An arbitrary {@code base -> shocked} perturbation of {@code input}. */
  public static Shock custom(String input, DoubleUnaryOperator transform) {
    return new Shock(input, Kind.CUSTOM, Double.NaN, transform);
  }

  /** The shocked value of {@code input}, given its base value. */
  public double shocked(double base) {
    return switch (kind) {
      case ABSOLUTE -> amount;
      case RELATIVE -> base * (1.0 + amount);
      case ADDITIVE -> base + amount;
      case CUSTOM -> transform.applyAsDouble(base);
    };
  }
}
