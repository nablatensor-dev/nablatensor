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
package com.nablatensor.tensor.spi;

import com.nablatensor.tensor.Op;
import com.nablatensor.tensor.expr.Expr;

import java.util.IdentityHashMap;
import java.util.Map;

/**
 * The compute-kernel source shared by the CUDA and ROCm/HIP tensor backends.
 *
 * <p>The two backends run byte-for-byte the same kernels: the bodies are plain
 * CUDA C that NVRTC compiles to PTX and HIPRTC accepts unchanged and compiles to
 * a GCN code object. Keeping the source here rather than copied into each backend
 * means an op added or a bug fixed lands on both GPU flavours at once. (The
 * Vulkan backend is unrelated — it runs GLSL compute shaders through shaderc.)
 *
 * <p>If a kernel ever needs architecture-specific tuning that CUDA C's shared
 * subset cannot express (wavefront-64 vs warp-32 reductions, differing block
 * sizes, {@code __syncwarp}), split that one kernel back out into the backend
 * that needs the variant; everything that stays identical stays here.
 */
public final class GpuKernels {

  private GpuKernels() {
  }

  /** Entry-point name of the kernel {@link #fusedSource} generates. */
  public static final String FUSED_KERNEL_NAME = "fused_kernel";

  /** Kernels in {@link #TENSOR_SOURCE}, in a stable order for module loading. */
  public static final String[] TENSOR_KERNEL_NAMES = {
      "ew_binary", "ew_scalar", "ew_unary", "transpose2d", "matmul_tiled", "batched_matmul_tiled",
      "random_uniform", "random_normal",
      "reduce_sum", "reduce_max", "sum_axis0", "reduce_axis_sum", "reduce_axis_max",
      "reduce_axis_argmax", "reduce_axis_max_backward", "broadcast_to",
      "conv2d_fwd", "conv2d_dx", "conv2d_dw", "relu_backward"
  };

  /** Every static tensor kernel, as one CUDA-C translation unit. */
  public static final String TENSOR_SOURCE = """
      extern "C" __global__ void ew_binary(float* out, const float* a, const float* b, int n, int op) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        float x = a[i], y = b[i], r = 0.0f;
        switch (op) {
          case 0: r = x + y; break;
          case 1: r = x - y; break;
          case 2: r = x * y; break;
          case 3: r = x / y; break;
          case 4: r = isnan(x) ? x : (isnan(y) ? y : fmaxf(x, y)); break;
          case 5: r = isnan(x) ? x : (isnan(y) ? y : fminf(x, y)); break;
        }
        out[i] = r;
      }
      extern "C" __global__ void ew_scalar(float* out, const float* a, float s, int n, int op) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        float x = a[i], r = 0.0f;
        switch (op) {
          case 0: r = x + s; break;
          case 1: r = x - s; break;
          case 2: r = x * s; break;
          case 3: r = x / s; break;
          case 4: r = isnan(x) ? x : (isnan(s) ? s : fmaxf(x, s)); break;
          case 5: r = isnan(x) ? x : (isnan(s) ? s : fminf(x, s)); break;
        }
        out[i] = r;
      }
      extern "C" __global__ void ew_unary(float* out, const float* a, int n, int op) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        float x = a[i], r = 0.0f;
        switch (op) {
          case 0: r = -x; break;
          case 1: r = expf(x); break;
          case 2: r = logf(x); break;
          case 3: r = sqrtf(x); break;
          case 4: r = rsqrtf(x); break;
          case 5: r = tanhf(x); break;
          case 6: r = 1.0f / (1.0f + expf(-x)); break;
          case 7: r = isnan(x) ? x : fmaxf(0.0f, x); break;
          case 8: r = fabsf(x); break;
          case 9: r = x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); break;
        }
        out[i] = r;
      }
      extern "C" __global__ void transpose2d(float* out, const float* in, int rows, int cols) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= rows * cols) return;
        int r = i / cols, c = i % cols;
        out[c * rows + r] = in[i];
      }
      extern "C" __global__ void matmul_tiled(
          float* out, const float* a, const float* b, int M, int K, int N,
          unsigned long long tile_offset) {
        __shared__ float tile_a[16][16];
        __shared__ float tile_b[16][16];
        unsigned long long global_tile = tile_offset + (unsigned long long) blockIdx.x;
        unsigned long long column_tiles = ((unsigned long long) N + 15ULL) / 16ULL;
        unsigned long long tile_row = global_tile / column_tiles;
        unsigned long long tile_col = global_tile % column_tiles;
        unsigned long long row = tile_row * 16ULL + threadIdx.y;
        unsigned long long col = tile_col * 16ULL + threadIdx.x;
        float acc = 0.0f;
        unsigned long long k_tiles = ((unsigned long long) K + 15ULL) / 16ULL;
        for (unsigned long long tile = 0; tile < k_tiles; tile++) {
          unsigned long long a_col = tile * 16ULL + threadIdx.x;
          unsigned long long b_row = tile * 16ULL + threadIdx.y;
          tile_a[threadIdx.y][threadIdx.x] =
              row < (unsigned long long) M && a_col < (unsigned long long) K
                  ? a[row * (unsigned long long) K + a_col] : 0.0f;
          tile_b[threadIdx.y][threadIdx.x] =
              b_row < (unsigned long long) K && col < (unsigned long long) N
                  ? b[b_row * (unsigned long long) N + col] : 0.0f;
          __syncthreads();
          #pragma unroll
          for (int p = 0; p < 16; p++) {
            acc += tile_a[threadIdx.y][p] * tile_b[p][threadIdx.x];
          }
          __syncthreads();
        }
        if (row < (unsigned long long) M && col < (unsigned long long) N) {
          out[row * (unsigned long long) N + col] = acc;
        }
      }
      extern "C" __global__ void batched_matmul_tiled(
          float* out, const float* a, const float* b, int batch, int M, int K, int N,
          unsigned long long tile_offset) {
        __shared__ float tile_a[16][16];
        __shared__ float tile_b[16][16];
        unsigned long long global_tile = tile_offset + (unsigned long long) blockIdx.x;
        unsigned long long row_tiles = ((unsigned long long) M + 15ULL) / 16ULL;
        unsigned long long column_tiles = ((unsigned long long) N + 15ULL) / 16ULL;
        unsigned long long tiles_per_batch = row_tiles * column_tiles;
        unsigned long long batch_index = global_tile / tiles_per_batch;
        unsigned long long batch_tile = global_tile % tiles_per_batch;
        unsigned long long row = (batch_tile / column_tiles) * 16ULL + threadIdx.y;
        unsigned long long col = (batch_tile % column_tiles) * 16ULL + threadIdx.x;
        unsigned long long a_offset = batch_index * (unsigned long long) M * K;
        unsigned long long b_offset = batch_index * (unsigned long long) K * N;
        unsigned long long out_offset = batch_index * (unsigned long long) M * N;
        float acc = 0.0f;
        unsigned long long k_tiles = ((unsigned long long) K + 15ULL) / 16ULL;
        for (unsigned long long tile = 0; tile < k_tiles; tile++) {
          unsigned long long a_col = tile * 16ULL + threadIdx.x;
          unsigned long long b_row = tile * 16ULL + threadIdx.y;
          tile_a[threadIdx.y][threadIdx.x] =
              row < (unsigned long long) M && a_col < (unsigned long long) K
                  ? a[a_offset + row * (unsigned long long) K + a_col] : 0.0f;
          tile_b[threadIdx.y][threadIdx.x] =
              b_row < (unsigned long long) K && col < (unsigned long long) N
                  ? b[b_offset + b_row * (unsigned long long) N + col] : 0.0f;
          __syncthreads();
          #pragma unroll
          for (int p = 0; p < 16; p++) {
            acc += tile_a[threadIdx.y][p] * tile_b[p][threadIdx.x];
          }
          __syncthreads();
        }
        if (batch_index < (unsigned long long) batch
            && row < (unsigned long long) M && col < (unsigned long long) N) {
          out[out_offset + row * (unsigned long long) N + col] = acc;
        }
      }
      __device__ __forceinline__ unsigned long long random_mix64(unsigned long long value) {
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31);
      }
      __device__ __forceinline__ float random_uniform_value(
          unsigned long long seed, unsigned long long counter) {
        unsigned long long bits =
            random_mix64(seed + 0x9E3779B97F4A7C15ULL * counter);
        return (float) (bits >> 40) * 5.9604644775390625e-8f;
      }
      extern "C" __global__ void random_uniform(
          float* out, unsigned long long seed, unsigned long long counter, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = random_uniform_value(seed, counter + (unsigned long long) i);
      }
      extern "C" __global__ void random_normal(
          float* out, unsigned long long seed, unsigned long long counter, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        unsigned long long first = counter + 2ULL * (unsigned long long) i;
        unsigned long long bits = random_mix64(seed + 0x9E3779B97F4A7C15ULL * first);
        float u1 = (float) ((bits >> 40) + 1ULL) * 5.9604644775390625e-8f;
        float u2 = random_uniform_value(seed, first + 1ULL);
        out[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853071795864769f * u2);
      }
      extern "C" __global__ void reduce_sum(float* out, const float* a, int n) {
        __shared__ float sdata[256];
        int tid = threadIdx.x;
        float acc = 0.0f;
        for (int i = tid; i < n; i += blockDim.x) acc += a[i];
        sdata[tid] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
          if (tid < s) sdata[tid] += sdata[tid + s];
          __syncthreads();
        }
        if (tid == 0) out[0] = sdata[0];
      }
      extern "C" __global__ void reduce_max(float* out, const float* a, int n) {
        __shared__ float sdata[256];
        int tid = threadIdx.x;
        float best = -3.402823466e+38f;
        for (int i = tid; i < n; i += blockDim.x) { float v = a[i]; if (v > best) best = v; }
        sdata[tid] = best;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
          if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
          __syncthreads();
        }
        if (tid == 0) out[0] = sdata[0];
      }
      extern "C" __global__ void sum_axis0(float* out, const float* a, int rows, int cols) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= cols) return;
        float acc = 0.0f;
        for (int r = 0; r < rows; r++) acc += a[r * cols + c];
        out[c] = acc;
      }
      extern "C" __global__ void reduce_axis_sum(
          float* out, const float* a, int outer, int axis_size, int inner) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= outer * inner) return;
        int outer_index = index / inner;
        int inner_index = index % inner;
        int base = outer_index * axis_size * inner + inner_index;
        float total = 0.0f;
        for (int i = 0; i < axis_size; i++) total += a[base + i * inner];
        out[index] = total;
      }
      extern "C" __global__ void reduce_axis_max(
          float* out, const float* a, int outer, int axis_size, int inner) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= outer * inner) return;
        int outer_index = index / inner;
        int inner_index = index % inner;
        int base = outer_index * axis_size * inner + inner_index;
        float best = -1.0f / 0.0f;
        for (int i = 0; i < axis_size; i++) {
          float value = a[base + i * inner];
          if (isnan(value) || value > best) best = value;
        }
        out[index] = best;
      }
      extern "C" __global__ void reduce_axis_argmax(
          float* out, const float* a, int outer, int axis_size, int inner) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= outer * inner) return;
        int outer_index = index / inner;
        int inner_index = index % inner;
        int base = outer_index * axis_size * inner + inner_index;
        float best = -3.402823466e+38f;
        int best_index = 0;
        for (int i = 0; i < axis_size; i++) {
          float value = a[base + i * inner];
          if (value > best) {
            best = value;
            best_index = i;
          }
        }
        out[index] = (float) best_index;
      }
      extern "C" __global__ void reduce_axis_max_backward(
          float* out, const float* upstream, const float* input,
          int outer, int axis_size, int inner) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= outer * inner) return;
        int outer_index = index / inner;
        int inner_index = index % inner;
        int base = outer_index * axis_size * inner + inner_index;
        float best = -1.0f / 0.0f;
        int best_index = 0;
        for (int i = 0; i < axis_size; i++) {
          float value = input[base + i * inner];
          if (isnan(value) || value > best) {
            best = value;
            best_index = i;
            if (isnan(value)) break;
          }
        }
        for (int i = 0; i < axis_size; i++) {
          out[base + i * inner] = i == best_index ? upstream[index] : 0.0f;
        }
      }
      extern "C" __global__ void broadcast_to(float* out, const float* in,
          int d0, int d1, int d2, int d3, int s0, int s1, int s2, int s3, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int dims[4] = {d0, d1, d2, d3};
        int strides[4] = {s0, s1, s2, s3};
        long tstride[4];
        long acc = 1;
        for (int d = 3; d >= 0; d--) { tstride[d] = acc; acc *= dims[d]; }
        long rem = i;
        long srcOffset = 0;
        for (int d = 0; d < 4; d++) {
          long coord = rem / tstride[d];
          rem = rem % tstride[d];
          srcOffset += coord * strides[d];
        }
        out[i] = in[srcOffset];
      }
      extern "C" __global__ void conv2d_fwd(float* out, const float* x, const float* w,
          int batch, int inC, int inH, int inW, int outC, int k, int stride, int pad,
          int outH, int outW) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch * outC * outH * outW;
        if (index >= total) return;
        int ow = index % outW;
        int oh = (index / outW) % outH;
        int oc = (index / (outW * outH)) % outC;
        int image = index / (outW * outH * outC);
        int inputSize = inC * inH * inW;
        int weightSize = inC * k * k;
        float acc = 0.0f;
        for (int ic = 0; ic < inC; ic++) {
          const float* channel = x + image * inputSize + ic * inH * inW;
          const float* filter = w + oc * weightSize + ic * k * k;
          for (int kh = 0; kh < k; kh++) {
            int ih = oh * stride - pad + kh;
            if (ih < 0 || ih >= inH) continue;
            for (int kw = 0; kw < k; kw++) {
              int iw = ow * stride - pad + kw;
              if (iw < 0 || iw >= inW) continue;
              acc += channel[ih * inW + iw] * filter[kh * k + kw];
            }
          }
        }
        out[index] = acc;
      }
      extern "C" __global__ void conv2d_dx(float* dx, const float* dOut, const float* w,
          int batch, int inC, int inH, int inW, int outC, int k, int stride, int pad,
          int outH, int outW) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch * inC * inH * inW;
        if (index >= total) return;
        int iw = index % inW;
        int ih = (index / inW) % inH;
        int ic = (index / (inW * inH)) % inC;
        int image = index / (inW * inH * inC);
        int outputSize = outC * outH * outW;
        int weightSize = inC * k * k;
        float acc = 0.0f;
        for (int kh = 0; kh < k; kh++) {
          int top = ih + pad - kh;
          if (top < 0 || top % stride != 0) continue;
          int oh = top / stride;
          if (oh >= outH) continue;
          for (int kw = 0; kw < k; kw++) {
            int left = iw + pad - kw;
            if (left < 0 || left % stride != 0) continue;
            int ow = left / stride;
            if (ow >= outW) continue;
            for (int oc = 0; oc < outC; oc++) {
              float g = dOut[image * outputSize + (oc * outH + oh) * outW + ow];
              acc += g * w[oc * weightSize + ic * k * k + kh * k + kw];
            }
          }
        }
        dx[index] = acc;
      }
      extern "C" __global__ void conv2d_dw(float* dw, const float* x, const float* dOut,
          int batch, int inC, int inH, int inW, int outC, int k, int stride, int pad,
          int outH, int outW) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int weightSize = inC * k * k;
        if (index >= outC * weightSize) return;
        int kw = index % k;
        int kh = (index / k) % k;
        int ic = (index / (k * k)) % inC;
        int oc = index / weightSize;
        int inputSize = inC * inH * inW;
        int outputSize = outC * outH * outW;
        float acc = 0.0f;
        for (int image = 0; image < batch; image++) {
          const float* channel = x + image * inputSize + ic * inH * inW;
          const float* grad = dOut + image * outputSize + oc * outH * outW;
          for (int oh = 0; oh < outH; oh++) {
            int ih = oh * stride - pad + kh;
            if (ih < 0 || ih >= inH) continue;
            for (int ow = 0; ow < outW; ow++) {
              int iw = ow * stride - pad + kw;
              if (iw < 0 || iw >= inW) continue;
              acc += grad[oh * outW + ow] * channel[ih * inW + iw];
            }
          }
        }
        dw[index] = acc;
      }
      extern "C" __global__ void relu_backward(float* out, const float* upstream,
          const float* input, int n) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n) return;
        out[index] = input[index] > 0.0f ? upstream[index] : 0.0f;
      }
      """;

  /**
   * Emits a whole elementwise expression chain as one {@code fused_kernel}
   * translation unit, so a backend can run the chain in a single launch with no
   * per-op intermediate buffers. The text is CUDA C; HIPRTC takes it unchanged.
   */
  public static String fusedSource(Expr expr, int numInputs) {
    StringBuilder body = new StringBuilder();
    Map<Expr, String> names = new IdentityHashMap<>();
    String result = emitFused(expr, names, body);
    StringBuilder source = new StringBuilder(
        "extern \"C\" __global__ void " + FUSED_KERNEL_NAME + "(float* out");
    for (int i = 0; i < numInputs; i++) {
      source.append(", const float* in").append(i);
    }
    source.append(", int n) {\n")
        .append("  int i = blockIdx.x * blockDim.x + threadIdx.x;\n")
        .append("  if (i >= n) return;\n")
        .append(body)
        .append("  out[i] = ").append(result).append(";\n")
        .append("}\n");
    return source.toString();
  }

  /**
   * Emits one SSA-style local per unique node (shared subtrees reuse their
   * name), post-order, so a diamond in the expression tree is computed once. The
   * name is allocated from {@code names.size()} only after recursing into
   * children, so each node's number reflects how many distinct nodes were
   * already emitted.
   */
  private static String emitFused(Expr expr, Map<Expr, String> names, StringBuilder body) {
    String cached = names.get(expr);
    if (cached != null) {
      return cached;
    }
    String name;
    switch (expr) {
      case Expr.Input in -> {
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = in")
            .append(in.index()).append("[i];\n");
      }
      case Expr.Unary u -> {
        String x = emitFused(u.in(), names, body);
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = ").append(unaryExpr(u.op(), x)).append(";\n");
      }
      case Expr.Binary b -> {
        String l = emitFused(b.left(), names, body);
        String r = emitFused(b.right(), names, body);
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = ").append(binaryExpr(b.op(), l, r)).append(";\n");
      }
      case Expr.Scalar s -> {
        String x = emitFused(s.in(), names, body);
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = ")
            .append(binaryExpr(s.op(), x, literal(s.value()))).append(";\n");
      }
    }
    names.put(expr, name);
    return name;
  }

  private static String literal(double value) {
    return Float.toString((float) value) + "f";
  }

  private static String binaryExpr(Op op, String l, String r) {
    return switch (op) {
      case ADD -> "(" + l + " + " + r + ")";
      case SUB -> "(" + l + " - " + r + ")";
      case MUL -> "(" + l + " * " + r + ")";
      case DIV -> "(" + l + " / " + r + ")";
      case MAX -> "(isnan(" + l + ") ? (" + l + ") : (isnan(" + r + ") ? (" + r
          + ") : fmaxf(" + l + ", " + r + ")))";
      case MIN -> "(isnan(" + l + ") ? (" + l + ") : (isnan(" + r + ") ? (" + r
          + ") : fminf(" + l + ", " + r + ")))";
      default -> throw new IllegalArgumentException("not a binary/scalar op: " + op);
    };
  }

  private static String unaryExpr(Op op, String x) {
    return switch (op) {
      case NEG -> "(-" + x + ")";
      case EXP -> "expf(" + x + ")";
      case LOG -> "logf(" + x + ")";
      case SQRT -> "sqrtf(" + x + ")";
      case RSQRT -> "rsqrtf(" + x + ")";
      case TANH -> "tanhf(" + x + ")";
      case SIGMOID -> "(1.0f / (1.0f + expf(-" + x + ")))";
      case RELU -> "(isnan(" + x + ") ? (" + x + ") : fmaxf(0.0f, " + x + "))";
      case ABS -> "fabsf(" + x + ")";
      case SIGN -> "((" + x + ") > 0.0f ? 1.0f : ((" + x + ") < 0.0f ? -1.0f : 0.0f))";
      default -> throw new IllegalArgumentException("not a unary op: " + op);
    };
  }
}
