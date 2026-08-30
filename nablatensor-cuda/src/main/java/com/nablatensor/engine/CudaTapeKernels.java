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

import java.util.EnumMap;
import java.util.Map;

/**
 * The tape-independent CUDA kernels, compiled once per JVM instead of once per
 * tape.
 *
 * <p>This is the whole point of the engines that use them. {@link
 * CudaAadCodegen} emits a bespoke kernel per tape — 130 KB of fully unrolled
 * source for a 1,534-node tape, which NVRTC takes seconds to compile — and that
 * cost is paid again for every new tape shape. The kernels here read the tape
 * as <em>data</em>, so their source never changes, compiles in a fraction of a
 * second and is reused for every tape thereafter.
 *
 * <p>What that costs is registers. A generated kernel indexes its values by
 * literal name, so the register allocator can keep them in registers; an
 * interpreter indexes by a runtime node number, which forces the value and
 * adjoint arrays into memory. These kernels are therefore bandwidth bound where
 * the generated one is occupancy bound.
 */
final class CudaTapeKernels {

  static final int BLOCK = 256;

  /** Fixed ceiling so a thread can hold its gradient accumulators in registers. */
  static final int MAX_INPUTS = 32;

  private static final Map<AadOptions.Precision, Handles> CACHE =
      new EnumMap<>(AadOptions.Precision.class);

  record Handles(long interp, long eagerForward, long eagerReverse, long reduce,
                 double compileSeconds) {
  }

  private CudaTapeKernels() {
  }

  static synchronized Handles get(AadOptions.Precision precision) {
    Handles cached = CACHE.get(precision);
    if (cached != null) {
      // Reported as zero so a caller timing "compilation" sees what the second
      // and every later tape actually pays, which is nothing.
      return new Handles(cached.interp(), cached.eagerForward(), cached.eagerReverse(),
          cached.reduce(), 0.0);
    }
    String source = source(precision == AadOptions.Precision.FLOAT32);
    long start = System.nanoTime();
    long interp = CudaJit.compile(source, "aad_interp");
    long forward = CudaJit.compile(source, "eager_forward");
    long reverse = CudaJit.compile(source, "eager_reverse");
    long reduce = CudaJit.compile(source, "reduce_sum");
    double seconds = (System.nanoTime() - start) / 1e9;
    Handles handles = new Handles(interp, forward, reverse, reduce, seconds);
    CACHE.put(precision, handles);
    return handles;
  }

  private static String source(boolean f32) {
    String real = f32 ? "float" : "double";
    String body = TEMPLATE
        .replace("REAL", real)
        .replace("RSQRT", f32 ? "sqrtf" : "sqrt")
        .replace("RLOG", f32 ? "logf" : "log")
        .replace("REXP", f32 ? "__expf" : "exp")
        .replace("RABS", f32 ? "fabsf" : "fabs")
        .replace("RMAX", f32 ? "fmaxf" : "fmax")
        .replace("RMIN", f32 ? "fminf" : "fmin")
        .replace("RSIN", f32 ? "__sinf" : "sin")
        .replace("RCOS", f32 ? "__cosf" : "cos");
    return "#define BLOCK " + BLOCK + "\n#define MAX_INPUTS " + MAX_INPUTS + "\n" + body;
  }

  // Opcode numbers are AadOp ordinals; see opcode().
  static int opcode(AadOp op) {
    return op.ordinal();
  }

  private static final String TEMPLATE = """
      typedef REAL real;

      #define OP_CONST 0
      #define OP_INPUT 1
      #define OP_RANDN 2
      #define OP_ADD   3
      #define OP_SUB   4
      #define OP_MUL   5
      #define OP_DIV   6
      #define OP_NEG   7
      #define OP_EXP   8
      #define OP_LOG   9
      #define OP_SQRT  10
      #define OP_MAX   11
      #define OP_MIN   12
      #define OP_ABS   13

      // Same counter-based stream as the generated kernel: draw k is Philox
      // counter k/2, cosine half when k is even and sine half when it is odd.
      __device__ __forceinline__ real philox_normal(
          unsigned long long path, unsigned long long seed, int drawIndex) {
        unsigned int c0 = (unsigned int) (path & 0xffffffffull) ^ (unsigned int) (seed & 0xffffffffull);
        unsigned int c1 = ((unsigned int) (path >> 32) ^ (unsigned int) (seed >> 32))
                          ^ (unsigned int) (drawIndex >> 1);
        unsigned int key = 0x1BD11BDAu;
        #pragma unroll
        for (int i = 0; i < 10; i++) {
          unsigned int hi = __umulhi(0xD256D193u, c0);
          unsigned int lo = 0xD256D193u * c0;
          c0 = hi ^ key ^ c1;
          c1 = lo;
          key += 0x9E3779B9u;
        }
        real u1 = ((real) c0 + (real) 0.5) * (real) 2.3283064365386963e-10;
        real u2 = ((real) c1 + (real) 0.5) * (real) 2.3283064365386963e-10;
        real radius = RSQRT(-(real) 2.0 * RLOG(u1));
        real angle = (real) 6.283185307179586 * u2;
        return (drawIndex & 1) ? radius * RSIN(angle) : radius * RCOS(angle);
      }

      // ---------------------------------------------------------------- interpreter

      extern "C" __global__ void aad_interp(
          const int* __restrict__ nodes,        // op, argA, argB, active per node
          const double* __restrict__ consts,
          const double* __restrict__ inputs,
          const int* __restrict__ inputNodes,
          const int* __restrict__ meta,         // nNodes, nInputs, outNode, adjoints
          unsigned long long nPaths,
          unsigned long long pathOffset,
          unsigned long long seed,
          real* __restrict__ vs,
          real* __restrict__ ds,
          double* __restrict__ partials) {

        const int nNodes = meta[0];
        const int nInputs = meta[1];
        const int outNode = meta[2];
        const int adjoints = meta[3];

        const unsigned long long tid =
            (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long threads =
            (unsigned long long) gridDim.x * blockDim.x;

        double accValue = 0.0;
        double accAdj[MAX_INPUTS];
        for (int j = 0; j < nInputs; j++) {
          accAdj[j] = 0.0;
        }

        for (unsigned long long path = tid; path < nPaths; path += threads) {
          const unsigned long long scenario = path + pathOffset;

          for (int i = 0; i < nNodes; i++) {
            const int op = nodes[i * 4];
            const int a = nodes[i * 4 + 1];
            const int b = nodes[i * 4 + 2];
            real result;
            switch (op) {
              case OP_CONST: result = (real) consts[i]; break;
              case OP_INPUT: result = (real) inputs[a]; break;
              case OP_RANDN: result = philox_normal(scenario, seed, a); break;
              case OP_ADD:   result = vs[(long) a * threads + tid] + vs[(long) b * threads + tid]; break;
              case OP_SUB:   result = vs[(long) a * threads + tid] - vs[(long) b * threads + tid]; break;
              case OP_MUL:   result = vs[(long) a * threads + tid] * vs[(long) b * threads + tid]; break;
              case OP_DIV:   result = vs[(long) a * threads + tid] / vs[(long) b * threads + tid]; break;
              case OP_NEG:   result = -vs[(long) a * threads + tid]; break;
              case OP_EXP:   result = REXP(vs[(long) a * threads + tid]); break;
              case OP_LOG:   result = RLOG(vs[(long) a * threads + tid]); break;
              case OP_SQRT:  result = RSQRT(vs[(long) a * threads + tid]); break;
              case OP_ABS:   result = RABS(vs[(long) a * threads + tid]); break;
              case OP_MAX:   result = RMAX(vs[(long) a * threads + tid], vs[(long) b * threads + tid]); break;
              case OP_MIN:   result = RMIN(vs[(long) a * threads + tid], vs[(long) b * threads + tid]); break;
              default:       result = (real) 0; break;
            }
            vs[(long) i * threads + tid] = result;
          }
          accValue += (double) vs[(long) outNode * threads + tid];

          if (!adjoints) {
            continue;
          }
          for (int i = 0; i < nNodes; i++) {
            ds[(long) i * threads + tid] = (real) 0;
          }
          ds[(long) outNode * threads + tid] = (real) 1;

          for (int i = nNodes - 1; i >= 0; i--) {
            if (!nodes[i * 4 + 3]) {
              continue;
            }
            const int op = nodes[i * 4];
            const int a = nodes[i * 4 + 1];
            const int b = nodes[i * 4 + 2];
            const real adj = ds[(long) i * threads + tid];
            switch (op) {
              case OP_ADD:
                ds[(long) a * threads + tid] += adj;
                ds[(long) b * threads + tid] += adj;
                break;
              case OP_SUB:
                ds[(long) a * threads + tid] += adj;
                ds[(long) b * threads + tid] -= adj;
                break;
              case OP_MUL:
                ds[(long) a * threads + tid] += adj * vs[(long) b * threads + tid];
                ds[(long) b * threads + tid] += adj * vs[(long) a * threads + tid];
                break;
              case OP_DIV: {
                const real den = vs[(long) b * threads + tid];
                ds[(long) a * threads + tid] += adj / den;
                ds[(long) b * threads + tid] -= adj * vs[(long) i * threads + tid] / den;
                break;
              }
              case OP_NEG:
                ds[(long) a * threads + tid] -= adj;
                break;
              case OP_EXP:
                ds[(long) a * threads + tid] += adj * vs[(long) i * threads + tid];
                break;
              case OP_LOG:
                ds[(long) a * threads + tid] += adj / vs[(long) a * threads + tid];
                break;
              case OP_SQRT:
                ds[(long) a * threads + tid] += adj * (real) 0.5 / vs[(long) i * threads + tid];
                break;
              case OP_ABS:
                ds[(long) a * threads + tid] +=
                    vs[(long) a * threads + tid] < (real) 0 ? -adj : adj;
                break;
              case OP_MAX:
                if (vs[(long) a * threads + tid] >= vs[(long) b * threads + tid]) {
                  ds[(long) a * threads + tid] += adj;
                } else {
                  ds[(long) b * threads + tid] += adj;
                }
                break;
              case OP_MIN:
                if (vs[(long) a * threads + tid] <= vs[(long) b * threads + tid]) {
                  ds[(long) a * threads + tid] += adj;
                } else {
                  ds[(long) b * threads + tid] += adj;
                }
                break;
              default:
                break;
            }
          }
          for (int j = 0; j < nInputs; j++) {
            accAdj[j] += (double) ds[(long) inputNodes[j] * threads + tid];
          }
        }

        const int channels = adjoints ? nInputs + 1 : 1;
        __shared__ double sh[BLOCK];
        for (int c = 0; c < channels; c++) {
          sh[threadIdx.x] = (c == 0) ? accValue : accAdj[c - 1];
          __syncthreads();
          for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
          }
          if (threadIdx.x == 0) partials[blockIdx.x * channels + c] = sh[0];
          __syncthreads();
        }
      }

      // ------------------------------------------------------------ eager, per node

      extern "C" __global__ void eager_forward(
          int op, int constIndex, int drawIndex, int inputIndex,
          unsigned long long pathBase, unsigned long long seed,
          const double* __restrict__ consts,
          const double* __restrict__ inputs,
          const real* __restrict__ va, const real* __restrict__ vb,
          real* __restrict__ out, unsigned long long n) {
        unsigned long long i = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long long stride = (unsigned long long) gridDim.x * blockDim.x;
        for (; i < n; i += stride) {
          real result;
          switch (op) {
            case OP_CONST: result = (real) consts[constIndex]; break;
            case OP_INPUT: result = (real) inputs[inputIndex]; break;
            case OP_RANDN: result = philox_normal(pathBase + i, seed, drawIndex); break;
            case OP_ADD:   result = va[i] + vb[i]; break;
            case OP_SUB:   result = va[i] - vb[i]; break;
            case OP_MUL:   result = va[i] * vb[i]; break;
            case OP_DIV:   result = va[i] / vb[i]; break;
            case OP_NEG:   result = -va[i]; break;
            case OP_EXP:   result = REXP(va[i]); break;
            case OP_LOG:   result = RLOG(va[i]); break;
            case OP_SQRT:  result = RSQRT(va[i]); break;
            case OP_ABS:   result = RABS(va[i]); break;
            case OP_MAX:   result = RMAX(va[i], vb[i]); break;
            case OP_MIN:   result = RMIN(va[i], vb[i]); break;
            default:       result = (real) 0; break;
          }
          out[i] = result;
        }
      }

      extern "C" __global__ void eager_reverse(
          int op,
          const real* __restrict__ va, const real* __restrict__ vb,
          const real* __restrict__ vi, const real* __restrict__ di,
          real* __restrict__ da, real* __restrict__ db,
          unsigned long long n) {
        unsigned long long i = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long long stride = (unsigned long long) gridDim.x * blockDim.x;
        for (; i < n; i += stride) {
          const real adj = di[i];
          switch (op) {
            case OP_ADD:
              if (da) da[i] += adj;
              if (db) db[i] += adj;
              break;
            case OP_SUB:
              if (da) da[i] += adj;
              if (db) db[i] -= adj;
              break;
            case OP_MUL:
              if (da) da[i] += adj * vb[i];
              if (db) db[i] += adj * va[i];
              break;
            case OP_DIV:
              if (da) da[i] += adj / vb[i];
              if (db) db[i] -= adj * vi[i] / vb[i];
              break;
            case OP_NEG:
              if (da) da[i] -= adj;
              break;
            case OP_EXP:
              if (da) da[i] += adj * vi[i];
              break;
            case OP_LOG:
              if (da) da[i] += adj / va[i];
              break;
            case OP_SQRT:
              if (da) da[i] += adj * (real) 0.5 / vi[i];
              break;
            case OP_ABS:
              if (da) da[i] += va[i] < (real) 0 ? -adj : adj;
              break;
            case OP_MAX: {
              bool takeA = va[i] >= vb[i];
              if (da && takeA) da[i] += adj;
              if (db && !takeA) db[i] += adj;
              break;
            }
            case OP_MIN: {
              bool takeA = va[i] <= vb[i];
              if (da && takeA) da[i] += adj;
              if (db && !takeA) db[i] += adj;
              break;
            }
            default:
              break;
          }
        }
      }

      extern "C" __global__ void reduce_sum(
          const real* __restrict__ x, unsigned long long n, double* __restrict__ partials) {
        __shared__ double sh[BLOCK];
        double sum = 0.0;
        unsigned long long i = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long long stride = (unsigned long long) gridDim.x * blockDim.x;
        for (; i < n; i += stride) {
          sum += (double) x[i];
        }
        sh[threadIdx.x] = sum;
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
          if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
          __syncthreads();
        }
        if (threadIdx.x == 0) partials[blockIdx.x] = sh[0];
      }
      """;
}
