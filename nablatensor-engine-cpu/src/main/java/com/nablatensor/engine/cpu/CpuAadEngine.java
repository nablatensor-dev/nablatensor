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
package com.nablatensor.engine.cpu;

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadExecutable;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadTape;

/**
 * Scalar JVM engine. Lowest priority, so it runs only when nothing faster is
 * present or when it is asked for by name — which is exactly what a
 * verification run wants, since it shares no code with the generated kernels.
 */
public final class CpuAadEngine implements AadEngine {

  @Override
  public String name() {
    return "cpu";
  }

  @Override
  public int priority() {
    return 10;
  }

  @Override
  public boolean isAvailable() {
    return true;
  }

  @Override
  public boolean supports(AadOptions options) {
    return options.precision() == AadOptions.Precision.FLOAT64;
  }

  @Override
  public String describe() {
    return "scalar JVM · " + Runtime.getRuntime().availableProcessors() + " processors · fp64";
  }

  @Override
  public AadExecutable compile(AadTape tape, AadOptions options) {
    if (!supports(options)) {
      throw new IllegalArgumentException("the scalar JVM engine is double-precision only");
    }
    return new ScalarReplay(tape, options);
  }
}
