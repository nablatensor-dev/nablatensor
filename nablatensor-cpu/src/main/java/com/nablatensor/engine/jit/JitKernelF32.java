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
package com.nablatensor.engine.jit;

/**
 * Single-precision counterpart of {@link JitKernel}. Arithmetic runs in
 * {@code float}; the transcendentals still go through {@code Math} in
 * {@code double} and narrow, as the JVM has no {@code float} {@code exp}/
 * {@code log}. Per-scenario value and gradient totals are accumulated by
 * {@link JitReplay} in {@code double}.
 */
public interface JitKernelF32 {

  float forward(float[] v, float[] in, float[] draws, float[] scratch);

  void reverse(float[] v, float[] d, float[] scratch, float[] draws);
}
