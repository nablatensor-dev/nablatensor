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
package com.nablatensor.engine.opencl;

import com.nablatensor.backend.opencl.OpenClCompute;
import com.nablatensor.engine.AadCheckpointPlan;
import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadExecutable;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.CudaAadCodegen;
import com.nablatensor.engine.DeviceAadExecutable;

/**
 * OpenCL replay engine: the recorded tape becomes a fused forward+adjoint
 * OpenCL C kernel, built once with {@code clBuildProgram} and replayed one
 * scenario per work-item.
 *
 * <p>Priority sits <em>below</em> SIMD, so {@code .fastest()} never lands here:
 * the OpenCL device chosen by default is usually the vendor GPU runtime, which
 * on the integrated parts this repo runs on shares the display scheduler and
 * carries the same wedge risk as the ROCm path. Ask for it deliberately with
 * {@code -Dnablatensor.engine=opencl} / {@code .on("opencl")}, and pick a
 * specific ICD with {@code -Dnablatensor.opencl.platform=<substring>} (e.g.
 * {@code rusticl}, {@code portable}, {@code pocl}) when more than one is
 * installed.
 *
 * <p>Holds no reference to {@code libOpenCL} beyond {@link #isAvailable()},
 * which probes through the {@link OpenClCompute} facade, so merely enumerating
 * engines cannot fail where OpenCL is absent.
 */
public final class OpenClAadEngine implements AadEngine {

  @Override
  public String name() {
    return "opencl";
  }

  @Override
  public int priority() {
    return 45;
  }

  @Override
  public boolean isAvailable() {
    try {
      return OpenClCompute.isAvailable()
          && OpenClCompute.supportsFp64()
          && OpenClCompute.maxWorkGroupSize() >= CudaAadCodegen.BLOCK;
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
      return "OpenCL · " + OpenClCompute.describe();
    } catch (Throwable ignored) {
      return "OpenCL · no device";
    }
  }

  @Override
  public AadExecutable compile(AadTape tape, AadOptions options) {
    AadEngine.requireBasicRandom(tape, "opencl");
    AadEngine.requireSingleOutput(tape, "opencl");
    if (!OpenClCompute.isAvailable() || !OpenClCompute.supportsFp64()) {
      throw new IllegalStateException(
          "no OpenCL device with cl_khr_fp64 available for the AAD replay kernel");
    }
    return DeviceAadExecutable.compile(tape, options, "opencl", OpenClDeviceRuntime.INSTANCE,
        OpenClAadEngine::generateSource, maxLaunchSeconds());
  }

  private static String generateSource(AadTape tape, AadOptions options, AadCheckpointPlan plan) {
    return plan != null
        ? OpenClAadCodegen.generateCheckpointed(tape, options, plan)
        : OpenClAadCodegen.generate(tape, options);
  }

  /**
   * The default OpenCL device is usually a GPU that also drives the display, so
   * a dispatch is kept well below the seconds-range hang check. Override with
   * {@code -Dnablatensor.maxLaunchSeconds}.
   */
  private static double maxLaunchSeconds() {
    String override = System.getProperty("nablatensor.maxLaunchSeconds");
    return override != null ? Double.parseDouble(override) : 0.5;
  }
}
