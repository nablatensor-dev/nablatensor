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

/** Numeric element types supported by nablatensor tensors. */
public enum DType {
  /** 32-bit IEEE-754 float. */
  F32(4),
  /** 64-bit IEEE-754 float. */
  F64(8),
  /** 16-bit IEEE-754 half float. */
  F16(2),
  /** 8-bit signed integer. */
  I8(1),
  /** 32-bit signed integer. */
  I32(4),
  /** 64-bit signed integer. */
  I64(8),
  /** Boolean, stored in one byte. */
  BOOL(1);

  private final int bytes;

  DType(int bytes) {
    this.bytes = bytes;
  }

  public int byteSize() {
    return bytes;
  }

  public boolean isFloating() {
    return this == F32 || this == F64 || this == F16;
  }
}
