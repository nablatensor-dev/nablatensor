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

import com.nablatensor.engine.AadCheckpointPlan;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.CudaAadCodegen;

/**
 * Produces the OpenCL C replay kernel for a recorded tape.
 *
 * <p>Rather than maintain a second line-for-line code generator, this reuses
 * {@link CudaAadCodegen} — whose output the ROCm engine already consumes
 * unchanged through HIPRTC — and rewrites the handful of constructs where
 * OpenCL C and CUDA C genuinely differ: the kernel qualifier and pointer
 * address spaces, the thread-index built-ins ({@code get_global_id} for
 * {@code blockIdx.x * blockDim.x + threadIdx.x} and friends), {@code __local}
 * for {@code __shared__}, {@code barrier()} for {@code __syncthreads()},
 * {@code mul_hi} for {@code __umulhi}, the counter RNG's struct-by-reference
 * parameter (OpenCL C has no references), and {@code sincos}'s two-pointer CUDA
 * signature versus OpenCL's return-sine / one-pointer form.
 *
 * <p>The only performance-motivated change is {@link #nativeMath}: on the fp32
 * path the Box-Muller transcendentals are mapped to the device's hardware
 * {@code native_*} instructions, which is the dominant speed-up and stays well
 * inside Monte-Carlo error (verified bit-exact run-to-run and within tolerance
 * of the fp64 oracle). The kernel keeps its reduction accumulators and the
 * host-visible partials in {@code double} regardless of working precision — an
 * fp32-with-Kahan variant was tried and measured to be a wash on the target
 * APU — so the device must expose {@code cl_khr_fp64}; the engine checks that
 * before it offers to compile.
 */
final class OpenClAadCodegen {

  static final String KERNEL_NAME = CudaAadCodegen.KERNEL_NAME;
  static final int BLOCK = CudaAadCodegen.BLOCK;

  private OpenClAadCodegen() {
  }

  /**
   * Whether the fp32 kernel uses the hardware {@code native_*} transcendentals
   * for the RNG's Box-Muller pair and the per-step {@code exp}. On by default;
   * turn off with {@code -Dnablatensor.opencl.nativeMath=false} for a
   * correctly-rounded cross-check.
   */
  static boolean nativeMath() {
    return !"false".equalsIgnoreCase(System.getProperty("nablatensor.opencl.nativeMath", "true"));
  }

  static String generate(AadTape tape, AadOptions options) {
    boolean f32 = options.precision() == AadOptions.Precision.FLOAT32;
    return toOpenCl(CudaAadCodegen.generate(tape, options), f32);
  }

  static String generateCheckpointed(AadTape tape, AadOptions options, AadCheckpointPlan plan) {
    boolean f32 = options.precision() == AadOptions.Precision.FLOAT32;
    return toOpenCl(CudaAadCodegen.generateCheckpointed(tape, options, plan), f32);
  }

  /** Rewrites one generated CUDA-C replay kernel into the OpenCL C dialect. */
  private static String toOpenCl(String cuda, boolean f32) {
    String s = cuda;

    // 64-bit integers: OpenCL C spells the type `ulong` and always has it.
    s = s.replace("unsigned long long", "ulong");

    // Kernel signature and pointer address spaces.
    s = s.replace("extern \"C\" __global__ void ", "__kernel void ");
    s = s.replace("const double* __restrict__ inputs", "__global const double* restrict inputs");
    s = s.replace("double* __restrict__ partials", "__global double* restrict partials");
    s = s.replace("real* __restrict__ scratch", "__global real* restrict scratch");

    // Device helper qualifiers and the counter-RNG's by-reference parameter.
    s = s.replace("__device__ __forceinline__ ", "inline ");
    s = s.replace("struct Rng { unsigned int lo; unsigned int hi; };",
        "typedef struct { unsigned int lo; unsigned int hi; } Rng;");
    s = s.replace("Rng &g", "Rng* g");
    s = s.replace("g.lo", "g->lo").replace("g.hi", "g->hi");
    s = s.replace("rng_init(rng,", "rng_init(&rng,")
        .replace("rng_normal(rng,", "rng_normal(&rng,")
        .replace("rng_uniform(rng,", "rng_uniform(&rng,");
    s = s.replace("__umulhi(", "mul_hi(");
    // `#pragma unroll` on the Philox rounds is understood by every clang-derived
    // OpenCL compiler (AMD, Intel, PoCL); keep it as the CUDA/ROCm engines do.

    // Math built-ins: OpenCL C overloads the unsuffixed names for float and
    // double alike, so the fp32 code generator's `*f` / `__*f` spellings and
    // the two-pointer `sincos` are the only ones that need touching.
    s = s.replace("__expf(", "exp(")
        .replace("__sincosf", "sincos")
        .replace("sqrtf(", "sqrt(")
        .replace("logf(", "log(")
        .replace("fabsf(", "fabs(")
        .replace("fmaxf(", "fmax(")
        .replace("fminf(", "fmin(");
    s = s.replace("sincos((real) 6.283185307179586 * u2, &s, &c);",
        "s = sincos((real) 6.283185307179586 * u2, &c);");

    // Thread-index built-ins. The compound forms first, then the leftovers.
    s = s.replace("(ulong) blockIdx.x * blockDim.x + threadIdx.x", "get_global_id(0)");
    s = s.replace("blockDim.x * gridDim.x", "get_local_size(0) * get_num_groups(0)");
    s = s.replace("threadIdx.x", "get_local_id(0)");
    s = s.replace("blockIdx.x", "get_group_id(0)");

    // Shared memory and its barrier.
    s = s.replace("__shared__ ", "__local ");
    s = s.replace("__syncthreads()", "barrier(CLK_LOCAL_MEM_FENCE)");

    if (f32 && nativeMath()) {
      s = applyNativeMath(s);
    }
    return "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" + s;
  }

  /**
   * fp32 speed-up: map the RNG's Box-Muller transcendentals to the hardware
   * {@code native_*} instructions (v_exp_f32 / v_log_f32 / v_rsq_f32 / v_sin_f32
   * / v_cos_f32 on GCN/RDNA). The forward-sweep {@code exp} of each log-return
   * goes the same way. Accuracy is a few ULP rather than correctly-rounded,
   * which is well inside the Monte-Carlo error of the price it feeds — the
   * engine's parity test confirms the result still matches the fp64 oracle and
   * is bit-identical from one replay to the next.
   */
  private static String applyNativeMath(String s) {
    // the one two-output sincos, expanded to the two native ops
    s = s.replace("s = sincos((real) 6.283185307179586 * u2, &c);",
        "s = native_sin((real) 6.283185307179586 * u2);"
            + " c = native_cos((real) 6.283185307179586 * u2);");
    s = s.replace("sqrt(-(real) 2.0 * log(u1))", "native_sqrt(-(real) 2.0 * native_log(u1))");
    // every remaining exp(...) in the kernel is a forward-sweep log-return
    s = s.replace("exp(", "native_exp(");
    return s;
  }
}
