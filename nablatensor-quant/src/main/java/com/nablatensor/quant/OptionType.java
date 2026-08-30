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
package com.nablatensor.quant;

/** Call or put. */
public enum OptionType {
  CALL,
  PUT;

  /** {@code +1} for a call, {@code -1} for a put; the payoff sign on {@code (underlying - strike)}. */
  public double sign() {
    return this == CALL ? 1.0 : -1.0;
  }
}
