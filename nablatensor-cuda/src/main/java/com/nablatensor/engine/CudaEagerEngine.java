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

import com.nablatensor.backend.cuda.CudaJit;

/**
 * CUDA engine that launches one kernel per tape node, as an eager tensor
 * framework would. Also compilation-free, and the slowest of the three CUDA
 * engines by some margin; it is here as the baseline the other two are worth
 * measuring against.
 */
public final class CudaEagerEngine implements AadEngine {

  @Override
  public String name() {
    return "cuda-eager";
  }

  @Override
  public int priority() {
    return 80;
  }

  @Override
  public boolean isAvailable() {
    try {
      return CudaJit.isAvailable();
    } catch (Throwable ignored) {
      return false;
    }
  }

  @Override
  public boolean supports(AadOptions options) {
    return true;
  }

  @Override
  public String describe() {
    try {
      return "CUDA eager, one launch per node · " + CudaJit.deviceName();
    } catch (Throwable ignored) {
      return "CUDA eager · no device";
    }
  }

  @Override
  public AadExecutable compile(AadTape tape, AadOptions options) {
    AadEngine.requireBasicRandom(tape, "cuda-eager");
    AadEngine.requireSingleOutput(tape, "cuda-eager");
    return new CudaEagerExecutable(tape, options);
  }
}
