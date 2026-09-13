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
 * An active scalar. Valuation code is written against this exactly as it would
 * be against {@code double}; every operation appends a node to the tape of the
 * recorder that produced it instead of computing a number.
 */
public final class ADouble {

  private final AadRecorder recorder;
  final int node;

  ADouble(AadRecorder recorder, int node) {
    this.recorder = recorder;
    this.node = node;
  }

  private ADouble binary(AadOp op, ADouble other) {
    if (other.recorder != recorder) {
      throw new IllegalArgumentException("operands come from different recordings");
    }
    return recorder.node(op, node, other.node);
  }

  private ADouble unary(AadOp op) {
    return recorder.node(op, node, -1);
  }

  public ADouble add(ADouble other) {
    return binary(AadOp.ADD, other);
  }

  public ADouble sub(ADouble other) {
    return binary(AadOp.SUB, other);
  }

  public ADouble mul(ADouble other) {
    return binary(AadOp.MUL, other);
  }

  public ADouble div(ADouble other) {
    return binary(AadOp.DIV, other);
  }

  public ADouble add(double value) {
    return add(recorder.constant(value));
  }

  public ADouble sub(double value) {
    return sub(recorder.constant(value));
  }

  public ADouble mul(double value) {
    return mul(recorder.constant(value));
  }

  public ADouble div(double value) {
    return div(recorder.constant(value));
  }

  public ADouble neg() {
    return unary(AadOp.NEG);
  }

  public ADouble exp() {
    return unary(AadOp.EXP);
  }

  public ADouble log() {
    return unary(AadOp.LOG);
  }

  public ADouble sqrt() {
    return unary(AadOp.SQRT);
  }

  public ADouble abs() {
    return unary(AadOp.ABS);
  }

  public ADouble max(ADouble other) {
    return binary(AadOp.MAX, other);
  }

  public ADouble min(ADouble other) {
    return binary(AadOp.MIN, other);
  }

  public ADouble max(double value) {
    return max(recorder.constant(value));
  }

  public ADouble min(double value) {
    return min(recorder.constant(value));
  }
}
