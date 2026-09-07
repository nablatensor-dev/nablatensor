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

/** Scalar operations a recorded tape can hold. */
public enum AadOp {
  /** A compile-time constant baked into the tape. */
  CONST,
  /** A named input the caller binds a value to on each replay. */
  INPUT,
  /** A standard normal draw that varies per scenario; produced on-device by the replay kernel. */
  RANDN,
  /** A uniform draw on {@code [0, 1)} that varies per scenario; produced on-device by the replay kernel. */
  RANDU,
  /** Binary addition, {@code a + b}. */
  ADD,
  /** Binary subtraction, {@code a - b}. */
  SUB,
  /** Binary multiplication, {@code a * b}. */
  MUL,
  /** Binary division, {@code a / b}. */
  DIV,
  /** Unary negation, {@code -a}. */
  NEG,
  /** Natural exponential, {@code e^a}. */
  EXP,
  /** Natural logarithm, {@code ln a}. */
  LOG,
  /** Square root, {@code sqrt(a)}. */
  SQRT,
  /** Binary maximum, {@code max(a, b)}. */
  MAX,
  /** Binary minimum, {@code min(a, b)}. */
  MIN,
  /** Absolute value, {@code |a|}. */
  ABS
}
