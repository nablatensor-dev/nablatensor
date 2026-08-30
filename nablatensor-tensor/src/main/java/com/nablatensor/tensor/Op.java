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
  // binary elementwise
  ADD, SUB, MUL, DIV, MAX, MIN,
  // unary elementwise
  NEG, EXP, LOG, SQRT, RSQRT, TANH, SIGMOID, RELU, ABS, SIGN
}
