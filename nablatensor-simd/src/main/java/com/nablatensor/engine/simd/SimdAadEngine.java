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
package com.nablatensor.engine.simd;

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadExecutable;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadTape;

/**
 * SIMD engine, sitting between CUDA and the scalar fallback in priority.
 *
 * <p>Deliberately contains no reference to any {@code jdk.incubator.vector}
 * type: the Vector API is an incubating module that is absent from the module
 * graph unless the JVM was started with {@code --add-modules
 * jdk.incubator.vector}. Were the types named here, merely enumerating the
 * available engines on a JVM without that flag would fail to link this class.
 * Everything that touches the API therefore lives in {@link SimdSupport} and
 * {@link VectorReplay}, which are only resolved after {@link #isAvailable()}
 * has confirmed the module is present.
 */
public final class SimdAadEngine implements AadEngine {

  private static final String PROBE_CLASS = "jdk.incubator.vector.DoubleVector";

  @Override
  public String name() {
    return "simd";
  }

  @Override
  public int priority() {
    return 50;
  }

  @Override
  public boolean isAvailable() {
    try {
      Class.forName(PROBE_CLASS, false, SimdAadEngine.class.getClassLoader());
      return true;
    } catch (Throwable absent) {
      return false;
    }
  }

  @Override
  public boolean supports(AadOptions options) {
    return true;
  }

  @Override
  public String describe() {
    if (!isAvailable()) {
      return "Vector API absent (start the JVM with --add-modules jdk.incubator.vector)";
    }
    return "Vector API · " + SimdSupport.describe()
        + " · " + Runtime.getRuntime().availableProcessors() + " processors · fp32+fp64";
  }

  @Override
  public AadExecutable compile(AadTape tape, AadOptions options) {
    AadEngine.requireBasicRandom(tape, "simd");
    AadEngine.requireSingleOutput(tape, "simd");
    if (!isAvailable()) {
      throw new IllegalStateException(
          "the Vector API is not on the module path; start the JVM with "
              + "--add-modules jdk.incubator.vector");
    }
    return options.precision() == AadOptions.Precision.FLOAT32
        ? new VectorReplayF32(tape, options)
        : new VectorReplayF64(tape, options);
  }
}
