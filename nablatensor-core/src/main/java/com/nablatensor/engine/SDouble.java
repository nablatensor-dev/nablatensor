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
public final class SDouble {

  private final AadRecorder recorder;
  final int node;

  SDouble(AadRecorder recorder, int node) {
    this.recorder = recorder;
    this.node = node;
  }

  private SDouble binary(AadOp op, SDouble other) {
    if (other.recorder != recorder) {
      throw new IllegalArgumentException("operands come from different recordings");
    }
    return recorder.node(op, node, other.node);
  }

  private SDouble unary(AadOp op) {
    return recorder.node(op, node, -1);
  }

  public SDouble add(SDouble other) {
    return binary(AadOp.ADD, other);
  }

  public SDouble sub(SDouble other) {
    return binary(AadOp.SUB, other);
  }

  public SDouble mul(SDouble other) {
    return binary(AadOp.MUL, other);
  }

  public SDouble div(SDouble other) {
    return binary(AadOp.DIV, other);
  }

  public SDouble add(double value) {
    return add(recorder.constant(value));
  }

  public SDouble sub(double value) {
    return sub(recorder.constant(value));
  }

  public SDouble mul(double value) {
    return mul(recorder.constant(value));
  }

  public SDouble div(double value) {
    return div(recorder.constant(value));
  }

  public SDouble neg() {
    return unary(AadOp.NEG);
  }

  public SDouble exp() {
    return unary(AadOp.EXP);
  }

  public SDouble log() {
    return unary(AadOp.LOG);
  }

  public SDouble sqrt() {
    return unary(AadOp.SQRT);
  }

  public SDouble abs() {
    return unary(AadOp.ABS);
  }

  public SDouble max(SDouble other) {
    return binary(AadOp.MAX, other);
  }

  public SDouble min(SDouble other) {
    return binary(AadOp.MIN, other);
  }

  public SDouble max(double value) {
    return max(recorder.constant(value));
  }

  public SDouble min(double value) {
    return min(recorder.constant(value));
  }
}
