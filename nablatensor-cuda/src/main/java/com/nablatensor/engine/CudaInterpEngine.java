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
 * CUDA engine that interprets the tape instead of generating a kernel for it.
 *
 * <p>Priority sits below the generating engine, so it is used when asked for by
 * name — which is the right call whenever the tape changes often enough that
 * seconds of NVRTC per shape outweigh a slower replay.
 */
public final class CudaInterpEngine implements AadEngine {

  @Override
  public String name() {
    return "cuda-interp";
  }

  @Override
  public int priority() {
    return 90;
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
      return "CUDA interpreter · " + CudaJit.deviceName() + " · no per-tape compilation";
    } catch (Throwable ignored) {
      return "CUDA interpreter · no device";
    }
  }

  @Override
  public AadExecutable compile(AadTape tape, AadOptions options) {
    AadEngine.requireBasicRandom(tape, "cuda-interp");
    AadEngine.requireSingleOutput(tape, "cuda-interp");
    return new CudaInterpExecutable(tape, options);
  }
}
