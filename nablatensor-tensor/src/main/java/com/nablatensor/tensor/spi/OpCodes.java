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
package com.nablatensor.tensor.spi;

import com.nablatensor.tensor.Op;

/**
 * The integer selector the elementwise kernels' {@code switch (op)} expects. The
 * CUDA-C kernels in {@link ElementwiseKernels} (which HIPRTC also compiles) and
 * the GLSL shaders in the {@code nablatensor-backend-vulkan} shader library both
 * branch on these exact values, so every backend maps {@link Op} through here
 * instead of repeating the table — change a case label in a kernel and you
 * change it here, once.
 */
public final class OpCodes {

  private OpCodes() {
  }

  /** Selector for {@code ew_binary} / {@code ew_scalar}. */
  public static int binary(Op op) {
    return switch (op) {
      case ADD -> 0;
      case SUB -> 1;
      case MUL -> 2;
      case DIV -> 3;
      case MAX -> 4;
      case MIN -> 5;
      default -> throw new IllegalArgumentException("not a binary/scalar op: " + op);
    };
  }

  /** Selector for {@code ew_unary}. */
  public static int unary(Op op) {
    return switch (op) {
      case NEG -> 0;
      case EXP -> 1;
      case LOG -> 2;
      case SQRT -> 3;
      case RSQRT -> 4;
      case TANH -> 5;
      case SIGMOID -> 6;
      case RELU -> 7;
      case ABS -> 8;
      case SIGN -> 9;
      default -> throw new IllegalArgumentException("not a unary op: " + op);
    };
  }
}
