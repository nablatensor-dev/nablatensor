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
package com.nablatensor.tensor;

/** Primitive operation kinds dispatched to the compute backends. */
public enum Op {
  // ---- binary elementwise ----
  /** Elementwise addition, {@code a + b}. */
  ADD,
  /** Elementwise subtraction, {@code a - b}. */
  SUB,
  /** Elementwise multiplication, {@code a * b}. */
  MUL,
  /** Elementwise division, {@code a / b}. */
  DIV,
  /** Elementwise maximum, {@code max(a, b)}. */
  MAX,
  /** Elementwise minimum, {@code min(a, b)}. */
  MIN,

  // ---- unary elementwise ----
  /** Negation, {@code -a}. */
  NEG,
  /** Natural exponential, {@code e^a}. */
  EXP,
  /** Natural logarithm, {@code ln a}. */
  LOG,
  /** Square root, {@code sqrt(a)}. */
  SQRT,
  /** Reciprocal square root, {@code 1 / sqrt(a)}. */
  RSQRT,
  /** Hyperbolic tangent. */
  TANH,
  /** Logistic sigmoid, {@code 1 / (1 + e^-a)}. */
  SIGMOID,
  /** Rectified linear unit, {@code max(a, 0)}. */
  RELU,
  /** Absolute value, {@code |a|}. */
  ABS,
  /** Sign, {@code -1 / 0 / +1}. */
  SIGN
}
