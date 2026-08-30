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
package com.nablatensor.backend.vulkan;

import com.nablatensor.tensor.ConvSpec;
import com.nablatensor.tensor.DType;
import com.nablatensor.tensor.Device;
import com.nablatensor.tensor.DeviceType;
import com.nablatensor.tensor.Op;
import com.nablatensor.tensor.Shape;
import com.nablatensor.tensor.expr.Expr;
import com.nablatensor.tensor.spi.ComputeBackend;
import com.nablatensor.tensor.spi.DeviceBuffer;

import java.util.IdentityHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Vulkan compute backend. Math runs in GLSL compute shaders compiled to SPIR-V
 * at runtime with {@code libshaderc} and dispatched through the Vulkan loader
 * via {@link java.lang.foreign}; tensors stay resident in device memory
 * (host-visible on an APU) between operations.
 *
 * <p>Phase-7 first cut: elementwise / reductions / matmul / transpose /
 * broadcast / relu-backward are implemented. Convolution is not.
 */
public final class VulkanBackend implements ComputeBackend {

  private static final int BLOCK = 256;
  private static final int MAX_BROADCAST_RANK = 4;

  /** matmul micro-tile per invocation and the resulting 64x64 block tile per 16x16 workgroup. */
  private static final int MM_TM = 4;
  private static final int MM_TN = 4;
  private static final int MM_BM = 16 * MM_TM;
  private static final int MM_BN = 16 * MM_TN;

  /** Phase-1 workgroup count for two-phase reductions; also the partials-buffer length. */
  private static final int REDUCE_GROUPS = 256;

  private static final String COMMON = "#version 450\n";

  private static final String EW_BINARY = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer A { float a[]; };
      layout(std430, binding = 2) readonly buffer B { float b[]; };
      layout(push_constant) uniform P { uint n; uint op; } p;
      void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= p.n) return;
        float x = a[i], y = b[i], r = 0.0;
        switch (p.op) {
          case 0u: r = x + y; break;
          case 1u: r = x - y; break;
          case 2u: r = x * y; break;
          case 3u: r = x / y; break;
          case 4u: r = isnan(x) ? x : (isnan(y) ? y : max(x, y)); break;
          case 5u: r = isnan(x) ? x : (isnan(y) ? y : min(x, y)); break;
        }
        o[i] = r;
      }
      """;

  private static final String EW_SCALAR = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer A { float a[]; };
      layout(push_constant) uniform P { uint n; uint op; float s; } p;
      void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= p.n) return;
        float x = a[i], s = p.s, r = 0.0;
        switch (p.op) {
          case 0u: r = x + s; break;
          case 1u: r = x - s; break;
          case 2u: r = x * s; break;
          case 3u: r = x / s; break;
          case 4u: r = isnan(x) ? x : (isnan(s) ? s : max(x, s)); break;
          case 5u: r = isnan(x) ? x : (isnan(s) ? s : min(x, s)); break;
        }
        o[i] = r;
      }
      """;

  private static final String EW_UNARY = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer A { float a[]; };
      layout(push_constant) uniform P { uint n; uint op; } p;
      void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= p.n) return;
        float x = a[i], r = 0.0;
        switch (p.op) {
          case 0u: r = -x; break;
          case 1u: r = exp(x); break;
          case 2u: r = log(x); break;
          case 3u: r = sqrt(x); break;
          case 4u: r = inversesqrt(x); break;
          case 5u: r = tanh(x); break;
          case 6u: r = 1.0 / (1.0 + exp(-x)); break;
          case 7u: r = isnan(x) ? x : max(0.0, x); break;
          case 8u: r = abs(x); break;
          case 9u: r = x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); break;
        }
        o[i] = r;
      }
      """;

  private static final String TRANSPOSE = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer I { float inp[]; };
      layout(push_constant) uniform P { uint rows; uint cols; } p;
      void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= p.rows * p.cols) return;
        uint r = i / p.cols, c = i % p.cols;
        o[c * p.rows + r] = inp[i];
      }
      """;

  /**
   * {@code o(M,N) = a(M,K) @ b(K,N)}, row-major. 16x16 workgroup; each invocation
   * accumulates a {@value #MM_TM}x{@value #MM_TN} micro-tile of the output in
   * registers, so a 64x64 block tile is computed per workgroup with 16 FMAs per
   * pair of shared-memory loads (the old kernel did one FMA per two loads, one
   * output element per invocation). {@code As} is stored transposed so the
   * inner-loop reads are unit-stride; the cooperative loads over
   * {@code As}/{@code Bs} are exactly {@code 1024 / 256 = 4} elements per thread.
   */
  private static final String MATMUL = COMMON + """
      layout(local_size_x = 16, local_size_y = 16) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly  buffer A { float a[]; };
      layout(std430, binding = 2) readonly  buffer B { float b[]; };
      layout(push_constant) uniform P { uint M; uint K; uint N; } p;

      const uint BM = 64u;
      const uint BN = 64u;
      const uint BK = 16u;
      const uint TM = 4u;
      const uint TN = 4u;

      shared float As[BK * BM];   // As[k * BM + m]  (A tile, transposed into shared)
      shared float Bs[BK * BN];   // Bs[k * BN + n]

      void main() {
        uint tid  = gl_LocalInvocationID.y * 16u + gl_LocalInvocationID.x;
        uint rowBase = gl_WorkGroupID.y * BM;
        uint colBase = gl_WorkGroupID.x * BN;
        uint rowT = rowBase + gl_LocalInvocationID.y * TM;
        uint colT = colBase + gl_LocalInvocationID.x * TN;

        float acc[TM][TN];
        for (uint i = 0u; i < TM; i++)
          for (uint j = 0u; j < TN; j++)
            acc[i][j] = 0.0;

        for (uint k0 = 0u; k0 < p.K; k0 += BK) {
          for (uint i = 0u; i < 4u; i++) {
            uint idx = tid + i * 256u;             // [0, 1024)
            uint sk = idx & 15u;                   // K within tile  -> coalesced global read
            uint sm = idx >> 4u;                   // [0, 64) row within block
            uint gr = rowBase + sm;
            uint gk = k0 + sk;
            As[sk * BM + sm] = (gr < p.M && gk < p.K) ? a[gr * p.K + gk] : 0.0;
          }
          for (uint i = 0u; i < 4u; i++) {
            uint idx = tid + i * 256u;
            uint sn = idx & 63u;                   // [0, 64) col within block -> coalesced
            uint sk = idx >> 6u;                   // K within tile
            uint gk = k0 + sk;
            uint gc = colBase + sn;
            Bs[sk * BN + sn] = (gk < p.K && gc < p.N) ? b[gk * p.N + gc] : 0.0;
          }
          barrier();

          for (uint kk = 0u; kk < BK; kk++) {
            float rA[TM];
            float rB[TN];
            for (uint i = 0u; i < TM; i++) rA[i] = As[kk * BM + gl_LocalInvocationID.y * TM + i];
            for (uint j = 0u; j < TN; j++) rB[j] = Bs[kk * BN + gl_LocalInvocationID.x * TN + j];
            for (uint i = 0u; i < TM; i++)
              for (uint j = 0u; j < TN; j++)
                acc[i][j] += rA[i] * rB[j];
          }
          barrier();
        }

        for (uint i = 0u; i < TM; i++) {
          uint r = rowT + i;
          if (r >= p.M) continue;
          for (uint j = 0u; j < TN; j++) {
            uint c = colT + j;
            if (c < p.N) o[r * p.N + c] = acc[i][j];
          }
        }
      }
      """;

  private static final String REDUCE_SUM = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer A { float a[]; };
      layout(push_constant) uniform P { uint n; } p;
      shared float sdata[256];
      void main() {
        uint tid = gl_LocalInvocationID.x;
        float acc = 0.0;
        for (uint i = tid; i < p.n; i += 256u) acc += a[i];
        sdata[tid] = acc;
        barrier();
        for (uint s = 128u; s > 0u; s >>= 1u) {
          if (tid < s) sdata[tid] += sdata[tid + s];
          barrier();
        }
        if (tid == 0u) o[0] = sdata[0];
      }
      """;

  private static final String REDUCE_MAX = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer A { float a[]; };
      layout(push_constant) uniform P { uint n; } p;
      shared float sdata[256];
      void main() {
        uint tid = gl_LocalInvocationID.x;
        float best = -3.402823466e+38;
        for (uint i = tid; i < p.n; i += 256u) best = max(best, a[i]);
        sdata[tid] = best;
        barrier();
        for (uint s = 128u; s > 0u; s >>= 1u) {
          if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
          barrier();
        }
        if (tid == 0u) o[0] = sdata[0];
      }
      """;

  private static final String SUM_AXIS0 = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer A { float a[]; };
      layout(push_constant) uniform P { uint rows; uint cols; } p;
      void main() {
        uint c = gl_GlobalInvocationID.x;
        if (c >= p.cols) return;
        float acc = 0.0;
        for (uint r = 0u; r < p.rows; r++) acc += a[r * p.cols + c];
        o[c] = acc;
      }
      """;

  /**
   * Reduce a rank-N buffer along one axis, viewed as {@code outer × axisSize ×
   * inner}. One invocation per {@code outer*inner} output element walks the axis
   * with stride {@code inner}. Mirrors the CUDA / ROCm {@code reduce_axis_*}
   * kernels. {@code argmax} writes the winning index as a float.
   */
  private static final String REDUCE_AXIS_PUSH =
      "layout(push_constant) uniform P { uint outer; uint axisSize; uint inner; } p;\n";

  private static final String REDUCE_AXIS_SUM = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer A { float a[]; };
      """ + REDUCE_AXIS_PUSH + """
      void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= p.outer * p.inner) return;
        uint base = (index / p.inner) * p.axisSize * p.inner + (index % p.inner);
        float total = 0.0;
        for (uint i = 0u; i < p.axisSize; i++) total += a[base + i * p.inner];
        o[index] = total;
      }
      """;

  private static final String REDUCE_AXIS_MAX = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer A { float a[]; };
      """ + REDUCE_AXIS_PUSH + """
      void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= p.outer * p.inner) return;
        uint base = (index / p.inner) * p.axisSize * p.inner + (index % p.inner);
        float best = -3.402823466e+38;
        for (uint i = 0u; i < p.axisSize; i++) {
          float value = a[base + i * p.inner];
          if (isnan(value) || value > best) best = value;
        }
        o[index] = best;
      }
      """;

  private static final String REDUCE_AXIS_ARGMAX = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer A { float a[]; };
      """ + REDUCE_AXIS_PUSH + """
      void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= p.outer * p.inner) return;
        uint base = (index / p.inner) * p.axisSize * p.inner + (index % p.inner);
        float best = -3.402823466e+38;
        uint bestIndex = 0u;
        for (uint i = 0u; i < p.axisSize; i++) {
          float value = a[base + i * p.inner];
          if (value > best) { best = value; bestIndex = i; }
        }
        o[index] = float(bestIndex);
      }
      """;

  private static final String REDUCE_AXIS_MAX_BACKWARD = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer U { float up[]; };
      layout(std430, binding = 2) readonly buffer X { float inp[]; };
      """ + REDUCE_AXIS_PUSH + """
      void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= p.outer * p.inner) return;
        uint base = (index / p.inner) * p.axisSize * p.inner + (index % p.inner);
        float best = -3.402823466e+38;
        uint bestIndex = 0u;
        for (uint i = 0u; i < p.axisSize; i++) {
          float value = inp[base + i * p.inner];
          if (isnan(value) || value > best) {
            best = value;
            bestIndex = i;
            if (isnan(value)) break;
          }
        }
        for (uint i = 0u; i < p.axisSize; i++) {
          o[base + i * p.inner] = i == bestIndex ? up[index] : 0.0;
        }
      }
      """;

  private static final String RELU_BACKWARD = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer U { float up[]; };
      layout(std430, binding = 2) readonly buffer X { float inp[]; };
      layout(push_constant) uniform P { uint n; } p;
      void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= p.n) return;
        o[i] = inp[i] > 0.0 ? up[i] : 0.0;
      }
      """;

  /**
   * Phase 1 of a two-phase reduction: a grid of {@value #REDUCE_GROUPS}
   * workgroups, each grid-strides over its share of the input and writes one
   * partial. Phase 2 is the existing single-workgroup {@code reduce_sum} /
   * {@code reduce_max} run over the {@value #REDUCE_GROUPS}-element partials
   * buffer. Replaces the old single-workgroup pass, which had 256 threads
   * reduce the whole tensor.
   */
  private static final String REDUCE_SUM_P1 = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float partials[]; };
      layout(std430, binding = 1) readonly buffer A { float a[]; };
      layout(push_constant) uniform P { uint n; } p;
      shared float sdata[256];
      void main() {
        uint tid = gl_LocalInvocationID.x;
        uint stride = gl_NumWorkGroups.x * 256u;
        float acc = 0.0;
        for (uint i = gl_GlobalInvocationID.x; i < p.n; i += stride) acc += a[i];
        sdata[tid] = acc;
        barrier();
        for (uint s = 128u; s > 0u; s >>= 1u) {
          if (tid < s) sdata[tid] += sdata[tid + s];
          barrier();
        }
        if (tid == 0u) partials[gl_WorkGroupID.x] = sdata[0];
      }
      """;

  private static final String REDUCE_MAX_P1 = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float partials[]; };
      layout(std430, binding = 1) readonly buffer A { float a[]; };
      layout(push_constant) uniform P { uint n; } p;
      shared float sdata[256];
      void main() {
        uint tid = gl_LocalInvocationID.x;
        uint stride = gl_NumWorkGroups.x * 256u;
        float best = -3.402823466e+38;
        for (uint i = gl_GlobalInvocationID.x; i < p.n; i += stride) best = max(best, a[i]);
        sdata[tid] = best;
        barrier();
        for (uint s = 128u; s > 0u; s >>= 1u) {
          if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
          barrier();
        }
        if (tid == 0u) partials[gl_WorkGroupID.x] = sdata[0];
      }
      """;

  private static final String BROADCAST = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly buffer I { float inp[]; };
      layout(push_constant) uniform P { uint d0, d1, d2, d3, s0, s1, s2, s3, n; } p;
      void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= p.n) return;
        uint dims[4] = uint[4](p.d0, p.d1, p.d2, p.d3);
        uint strides[4] = uint[4](p.s0, p.s1, p.s2, p.s3);
        uint tstride[4];
        uint acc = 1u;
        for (int d = 3; d >= 0; d--) { tstride[d] = acc; acc *= dims[d]; }
        uint rem = i;
        uint srcOffset = 0u;
        for (int d = 0; d < 4; d++) {
          uint coord = rem / tstride[d];
          rem = rem % tstride[d];
          srcOffset += coord * strides[d];
        }
        o[i] = inp[srcOffset];
      }
      """;

  /**
   * Direct 2-D convolution over batched, channel-major rank-2 images (see
   * {@link ConvSpec}). One output element per invocation, mirroring the CUDA /
   * ROCm {@code conv2d_fwd} / {@code conv2d_dx} / {@code conv2d_dw} kernels.
   * Push block: 10 uints {batch,inC,inH,inW,outC,k,stride,pad,outH,outW}.
   */
  private static final String CONV_PUSH =
      "layout(push_constant) uniform P { uint batch, inC, inH, inW, outC, k, stride, pad, outH, outW; } p;\n";

  private static final String CONV2D_FWD = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly  buffer X { float x[]; };
      layout(std430, binding = 2) readonly  buffer W { float wt[]; };
      """ + CONV_PUSH + """
      void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= p.batch * p.outC * p.outH * p.outW) return;
        uint ow = index % p.outW;
        uint oh = (index / p.outW) % p.outH;
        uint oc = (index / (p.outW * p.outH)) % p.outC;
        uint image = index / (p.outW * p.outH * p.outC);
        uint inputSize = p.inC * p.inH * p.inW;
        uint weightSize = p.inC * p.k * p.k;
        float acc = 0.0;
        for (uint ic = 0u; ic < p.inC; ic++) {
          uint chBase = image * inputSize + ic * p.inH * p.inW;
          uint fBase = oc * weightSize + ic * p.k * p.k;
          for (uint kh = 0u; kh < p.k; kh++) {
            int ih = int(oh * p.stride) - int(p.pad) + int(kh);
            if (ih < 0 || ih >= int(p.inH)) continue;
            for (uint kw = 0u; kw < p.k; kw++) {
              int iw = int(ow * p.stride) - int(p.pad) + int(kw);
              if (iw < 0 || iw >= int(p.inW)) continue;
              acc += x[chBase + uint(ih) * p.inW + uint(iw)] * wt[fBase + kh * p.k + kw];
            }
          }
        }
        o[index] = acc;
      }
      """;

  private static final String CONV2D_DX = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer DX { float dx[]; };
      layout(std430, binding = 1) readonly  buffer DO { float dOut[]; };
      layout(std430, binding = 2) readonly  buffer W  { float wt[]; };
      """ + CONV_PUSH + """
      void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= p.batch * p.inC * p.inH * p.inW) return;
        uint iw = index % p.inW;
        uint ih = (index / p.inW) % p.inH;
        uint ic = (index / (p.inW * p.inH)) % p.inC;
        uint image = index / (p.inW * p.inH * p.inC);
        uint outputSize = p.outC * p.outH * p.outW;
        uint weightSize = p.inC * p.k * p.k;
        float acc = 0.0;
        for (uint kh = 0u; kh < p.k; kh++) {
          int top = int(ih) + int(p.pad) - int(kh);
          if (top < 0 || (uint(top) % p.stride) != 0u) continue;
          uint oh = uint(top) / p.stride;
          if (oh >= p.outH) continue;
          for (uint kw = 0u; kw < p.k; kw++) {
            int left = int(iw) + int(p.pad) - int(kw);
            if (left < 0 || (uint(left) % p.stride) != 0u) continue;
            uint ow = uint(left) / p.stride;
            if (ow >= p.outW) continue;
            for (uint oc = 0u; oc < p.outC; oc++) {
              float g = dOut[image * outputSize + (oc * p.outH + oh) * p.outW + ow];
              acc += g * wt[oc * weightSize + ic * p.k * p.k + kh * p.k + kw];
            }
          }
        }
        dx[index] = acc;
      }
      """;

  private static final String CONV2D_DW = COMMON + """
      layout(local_size_x = 256) in;
      layout(std430, binding = 0) writeonly buffer DW { float dw[]; };
      layout(std430, binding = 1) readonly  buffer X  { float x[]; };
      layout(std430, binding = 2) readonly  buffer DO { float dOut[]; };
      """ + CONV_PUSH + """
      void main() {
        uint index = gl_GlobalInvocationID.x;
        uint weightSize = p.inC * p.k * p.k;
        if (index >= p.outC * weightSize) return;
        uint kw = index % p.k;
        uint kh = (index / p.k) % p.k;
        uint ic = (index / (p.k * p.k)) % p.inC;
        uint oc = index / weightSize;
        uint inputSize = p.inC * p.inH * p.inW;
        uint outputSize = p.outC * p.outH * p.outW;
        float acc = 0.0;
        for (uint image = 0u; image < p.batch; image++) {
          uint chBase = image * inputSize + ic * p.inH * p.inW;
          uint gBase = image * outputSize + oc * p.outH * p.outW;
          for (uint oh = 0u; oh < p.outH; oh++) {
            int ih = int(oh * p.stride) - int(p.pad) + int(kh);
            if (ih < 0 || ih >= int(p.inH)) continue;
            for (uint ow = 0u; ow < p.outW; ow++) {
              int iw = int(ow * p.stride) - int(p.pad) + int(kw);
              if (iw < 0 || iw >= int(p.inW)) continue;
              acc += dOut[gBase + oh * p.outW + ow] * x[chBase + uint(ih) * p.inW + uint(iw)];
            }
          }
        }
        dw[index] = acc;
      }
      """;

  private volatile boolean initialized;

  /**
   * Generated fused-elementwise shader source -&gt; the pipeline name it was
   * registered under. Keyed by source text so two op chains that emit the same
   * GLSL share one compiled pipeline, and a shape change (element count is a
   * push constant, not baked into the source) never recompiles.
   */
  private final ConcurrentHashMap<String, String> fusedPipelines = new ConcurrentHashMap<>();

  private static final AtomicInteger FUSED_SEQ = new AtomicInteger();

  @Override
  public String name() {
    return "vulkan";
  }

  @Override
  public DeviceType deviceType() {
    return DeviceType.VULKAN;
  }

  @Override
  public boolean isAvailable() {
    try {
      return VulkanRuntime.probe();
    } catch (Throwable failure) {
      return false;
    }
  }

  @Override
  public int priority() {
    return 40;
  }

  public String deviceName() {
    return VulkanRuntime.deviceName();
  }

  private synchronized void ensureInitialized() {
    if (initialized) {
      return;
    }
    VulkanRuntime.init();
    VulkanRuntime.registerPipeline("ew_binary", EW_BINARY, 3);
    VulkanRuntime.registerPipeline("ew_scalar", EW_SCALAR, 2);
    VulkanRuntime.registerPipeline("ew_unary", EW_UNARY, 2);
    VulkanRuntime.registerPipeline("transpose2d", TRANSPOSE, 2);
    VulkanRuntime.registerPipeline("matmul", MATMUL, 3);
    VulkanRuntime.registerPipeline("reduce_sum", REDUCE_SUM, 2);
    VulkanRuntime.registerPipeline("reduce_max", REDUCE_MAX, 2);
    VulkanRuntime.registerPipeline("reduce_sum_p1", REDUCE_SUM_P1, 2);
    VulkanRuntime.registerPipeline("reduce_max_p1", REDUCE_MAX_P1, 2);
    VulkanRuntime.registerPipeline("sum_axis0", SUM_AXIS0, 2);
    VulkanRuntime.registerPipeline("reduce_axis_sum", REDUCE_AXIS_SUM, 2);
    VulkanRuntime.registerPipeline("reduce_axis_max", REDUCE_AXIS_MAX, 2);
    VulkanRuntime.registerPipeline("reduce_axis_argmax", REDUCE_AXIS_ARGMAX, 2);
    VulkanRuntime.registerPipeline("reduce_axis_max_backward", REDUCE_AXIS_MAX_BACKWARD, 3);
    VulkanRuntime.registerPipeline("relu_backward", RELU_BACKWARD, 3);
    VulkanRuntime.registerPipeline("broadcast_to", BROADCAST, 2);
    VulkanRuntime.registerPipeline("conv2d_fwd", CONV2D_FWD, 3);
    VulkanRuntime.registerPipeline("conv2d_dx", CONV2D_DX, 3);
    VulkanRuntime.registerPipeline("conv2d_dw", CONV2D_DW, 3);
    initialized = true;
  }

  // ---- data movement ---------------------------------------------------------

  @Override
  public DeviceBuffer upload(float[] data, Shape shape, DType dtype, Device device) {
    ensureInitialized();
    requireVulkanDevice(device);
    if (dtype != DType.F32) {
      throw new UnsupportedOperationException("Vulkan float upload supports F32 only, got " + dtype);
    }
    long[] handles = VulkanRuntime.alloc(Math.max(4L, (long) data.length * Float.BYTES));
    VulkanRuntime.writeFloats(handles[1], data);
    return new VulkanBuffer(handles[0], handles[1], shape, dtype, device);
  }

  @Override
  public float[] download(DeviceBuffer buffer) {
    ensureInitialized();
    VulkanRuntime.sync();
    VulkanBuffer vk = vk(buffer);
    return VulkanRuntime.readFloats(vk.memory, vk.count());
  }

  // ---- elementwise ---------------------------------------------------------

  @Override
  public DeviceBuffer binary(Op op, DeviceBuffer a, DeviceBuffer b) {
    ensureInitialized();
    VulkanBuffer left = vk(a);
    VulkanBuffer right = vk(b);
    int n = left.count();
    if (n != right.count()) {
      throw new IllegalArgumentException("shape mismatch: " + left.shape() + " vs " + right.shape());
    }
    VulkanBuffer out = alloc(left.shape());
    VulkanRuntime.dispatch("ew_binary", grid(n), 1, 1,
        new long[] {out.buffer, left.buffer, right.buffer}, new int[] {n, binaryCode(op)}, -1, 0f);
    return out;
  }

  @Override
  public DeviceBuffer scalar(Op op, DeviceBuffer a, double value) {
    ensureInitialized();
    VulkanBuffer left = vk(a);
    int n = left.count();
    VulkanBuffer out = alloc(left.shape());
    VulkanRuntime.dispatch("ew_scalar", grid(n), 1, 1,
        new long[] {out.buffer, left.buffer}, new int[] {n, binaryCode(op)}, 2, (float) value);
    return out;
  }

  @Override
  public DeviceBuffer unary(Op op, DeviceBuffer a) {
    ensureInitialized();
    VulkanBuffer left = vk(a);
    int n = left.count();
    VulkanBuffer out = alloc(left.shape());
    VulkanRuntime.dispatch("ew_unary", grid(n), 1, 1,
        new long[] {out.buffer, left.buffer}, new int[] {n, unaryCode(op)}, -1, 0f);
    return out;
  }

  // ---- fused elementwise -------------------------------------------------

  /**
   * Runs a whole elementwise {@link Expr} chain as one generated GLSL compute
   * shader (one dispatch, one intermediate buffer), instead of the SPI default
   * of one dispatch and one buffer per node. The shader is emitted with the
   * same SSA post-order walk as the CUDA backend, compiled to SPIR-V once by
   * {@code libshaderc}, and cached (as a registered pipeline) by its source.
   */
  @Override
  public DeviceBuffer fused(Expr expr, DeviceBuffer[] inputs) {
    ensureInitialized();
    if (inputs.length == 0) {
      throw new IllegalArgumentException("fused requires at least one input");
    }
    VulkanBuffer[] operands = new VulkanBuffer[inputs.length];
    int n = -1;
    for (int i = 0; i < inputs.length; i++) {
      operands[i] = vk(inputs[i]);
      if (i == 0) {
        n = operands[i].count();
      } else if (operands[i].count() != n) {
        throw new IllegalArgumentException("fused inputs must share element count");
      }
    }
    int numInputs = inputs.length;
    String source = generateFusedGlsl(expr, numInputs);
    String pipeline = fusedPipelines.computeIfAbsent(source, src -> {
      String name = "fused_" + FUSED_SEQ.getAndIncrement();
      VulkanRuntime.registerPipeline(name, src, numInputs + 1);
      return name;
    });
    VulkanBuffer out = alloc(operands[0].shape());
    long[] handles = new long[numInputs + 1];
    handles[0] = out.buffer;
    for (int i = 0; i < numInputs; i++) {
      handles[i + 1] = operands[i].buffer;
    }
    VulkanRuntime.dispatch(pipeline, grid(n), 1, 1, handles, new int[] {n}, -1, 0f);
    return out;
  }

  private static String generateFusedGlsl(Expr expr, int numInputs) {
    StringBuilder body = new StringBuilder();
    String result = emitFused(expr, new IdentityHashMap<>(), body);
    StringBuilder src = new StringBuilder(256);
    src.append("#version 450\n")
        .append("layout(local_size_x = 256) in;\n")
        .append("layout(std430, binding = 0) writeonly buffer O { float o[]; };\n");
    for (int i = 0; i < numInputs; i++) {
      src.append("layout(std430, binding = ").append(i + 1)
          .append(") readonly buffer I").append(i)
          .append(" { float in").append(i).append("[]; };\n");
    }
    src.append("layout(push_constant) uniform P { uint n; } p;\n")
        .append("void main() {\n")
        .append("  uint i = gl_GlobalInvocationID.x;\n")
        .append("  if (i >= p.n) return;\n")
        .append(body)
        .append("  o[i] = ").append(result).append(";\n")
        .append("}\n");
    return src.toString();
  }

  /**
   * Emits one SSA-style local per unique node (shared subtrees reuse their
   * name), post-order. The name is taken from {@code names.size()} only after
   * the children have been emitted, so each node's number reflects how many
   * distinct nodes were already written.
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
        body.append("  float ").append(name).append(" = in").append(in.index()).append("[i];\n");
      }
      case Expr.Unary u -> {
        String x = emitFused(u.in(), names, body);
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = ").append(glslUnary(u.op(), x)).append(";\n");
      }
      case Expr.Binary b -> {
        String l = emitFused(b.left(), names, body);
        String r = emitFused(b.right(), names, body);
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = ").append(glslBinary(b.op(), l, r)).append(";\n");
      }
      case Expr.Scalar s -> {
        String x = emitFused(s.in(), names, body);
        name = "v" + names.size();
        body.append("  float ").append(name).append(" = ")
            .append(glslBinary(s.op(), x, glslLiteral(s.value()))).append(";\n");
      }
    }
    names.put(expr, name);
    return name;
  }

  private static String glslLiteral(double value) {
    float f = (float) value;
    if (Float.isNaN(f)) {
      return "uintBitsToFloat(0x7fc00000u)";
    }
    if (f == Float.POSITIVE_INFINITY) {
      return "uintBitsToFloat(0x7f800000u)";
    }
    if (f == Float.NEGATIVE_INFINITY) {
      return "uintBitsToFloat(0xff800000u)";
    }
    return Float.toString(f);   // always contains '.' or an exponent -> a valid GLSL float
  }

  private static String glslBinary(Op op, String l, String r) {
    return switch (op) {
      case ADD -> "(" + l + " + " + r + ")";
      case SUB -> "(" + l + " - " + r + ")";
      case MUL -> "(" + l + " * " + r + ")";
      case DIV -> "(" + l + " / " + r + ")";
      case MAX -> "(isnan(" + l + ") ? (" + l + ") : (isnan(" + r + ") ? (" + r
          + ") : max(" + l + ", " + r + ")))";
      case MIN -> "(isnan(" + l + ") ? (" + l + ") : (isnan(" + r + ") ? (" + r
          + ") : min(" + l + ", " + r + ")))";
      default -> throw new IllegalArgumentException("not a binary/scalar op: " + op);
    };
  }

  private static String glslUnary(Op op, String x) {
    return switch (op) {
      case NEG -> "(-" + x + ")";
      case EXP -> "exp(" + x + ")";
      case LOG -> "log(" + x + ")";
      case SQRT -> "sqrt(" + x + ")";
      case RSQRT -> "inversesqrt(" + x + ")";
      case TANH -> "tanh(" + x + ")";
      case SIGMOID -> "(1.0 / (1.0 + exp(-(" + x + "))))";
      case RELU -> "(isnan(" + x + ") ? (" + x + ") : max(0.0, " + x + "))";
      case ABS -> "abs(" + x + ")";
      case SIGN -> "((" + x + ") > 0.0 ? 1.0 : ((" + x + ") < 0.0 ? -1.0 : 0.0))";
      default -> throw new IllegalArgumentException("not a unary op: " + op);
    };
  }

  // ---- linear algebra ----------------------------------------------------

  @Override
  public DeviceBuffer transpose(DeviceBuffer a) {
    ensureInitialized();
    VulkanBuffer left = vk(a);
    Shape shape = left.shape();
    if (shape.rank() != 2) {
      throw new IllegalStateException("transpose requires a rank-2 tensor, got " + shape);
    }
    int rows = shape.dim(0);
    int cols = shape.dim(1);
    VulkanBuffer out = alloc(shape.transposed());
    VulkanRuntime.dispatch("transpose2d", grid(rows * cols), 1, 1,
        new long[] {out.buffer, left.buffer}, new int[] {rows, cols}, -1, 0f);
    return out;
  }

  @Override
  public DeviceBuffer matmul(DeviceBuffer a, DeviceBuffer b) {
    ensureInitialized();
    VulkanBuffer left = vk(a);
    VulkanBuffer right = vk(b);
    Shape ls = left.shape();
    Shape rs = right.shape();
    requireMatchingPlacement(left, right);
    if (ls.rank() != 2 || rs.rank() != 2 || ls.dim(1) != rs.dim(0)) {
      throw new IllegalArgumentException("incompatible matmul shapes: " + ls + " x " + rs);
    }
    int m = ls.dim(0);
    int k = ls.dim(1);
    int n = rs.dim(1);
    if (m < 1 || k < 1 || n < 1) {
      throw new IllegalArgumentException("matmul dimensions must be positive");
    }
    VulkanBuffer out = alloc(Shape.of(m, n));
    VulkanRuntime.dispatch("matmul", ceilDiv(n, MM_BN), ceilDiv(m, MM_BM), 1,
        new long[] {out.buffer, left.buffer, right.buffer}, new int[] {m, k, n}, -1, 0f);
    return out;
  }

  // ---- reductions ------------------------------------------------------------

  @Override
  public DeviceBuffer reduceSum(DeviceBuffer a) {
    return reduce(vk(a), "reduce_sum_p1", "reduce_sum");
  }

  @Override
  public DeviceBuffer reduceMax(DeviceBuffer a) {
    return reduce(vk(a), "reduce_max_p1", "reduce_max");
  }

  /**
   * Two-phase whole-tensor reduction: phase 1 spreads the input over
   * {@value #REDUCE_GROUPS} workgroups, each emitting one partial; phase 2 folds
   * the partials in a single workgroup. Small inputs (one phase-1 workgroup or
   * fewer) skip straight to the single-workgroup path.
   */
  private DeviceBuffer reduce(VulkanBuffer in, String phase1, String phase2) {
    ensureInitialized();
    int n = in.count();
    int groups = Math.min(REDUCE_GROUPS, Math.max(1, grid(n)));
    VulkanBuffer out = alloc(Shape.of(1));
    if (groups <= 1) {
      VulkanRuntime.dispatch(phase2, 1, 1, 1, new long[] {out.buffer, in.buffer}, new int[] {n}, -1, 0f);
      return out;
    }
    VulkanBuffer partials = alloc(Shape.of(groups));
    VulkanRuntime.dispatch(phase1, groups, 1, 1,
        new long[] {partials.buffer, in.buffer}, new int[] {n}, -1, 0f);
    VulkanRuntime.dispatch(phase2, 1, 1, 1,
        new long[] {out.buffer, partials.buffer}, new int[] {groups}, -1, 0f);
    partials.release();   // dispatch() blocks on vkQueueWaitIdle, so phase 2 is done reading
    return out;
  }

  @Override
  public DeviceBuffer sumAxis0(DeviceBuffer a) {
    ensureInitialized();
    VulkanBuffer in = vk(a);
    Shape shape = in.shape();
    if (shape.rank() != 2) {
      throw new IllegalStateException("sumAxis0 requires a rank-2 tensor, got " + shape);
    }
    int rows = shape.dim(0);
    int cols = shape.dim(1);
    VulkanBuffer out = alloc(Shape.of(cols));
    VulkanRuntime.dispatch("sum_axis0", grid(cols), 1, 1,
        new long[] {out.buffer, in.buffer}, new int[] {rows, cols}, -1, 0f);
    return out;
  }

  @Override
  public DeviceBuffer sumAxis(DeviceBuffer a, int axis, boolean keepDims) {
    return reduceAxis(a, axis, keepDims, "reduce_axis_sum");
  }

  @Override
  public DeviceBuffer maxAxis(DeviceBuffer a, int axis, boolean keepDims) {
    return reduceAxis(a, axis, keepDims, "reduce_axis_max");
  }

  @Override
  public DeviceBuffer argMaxAxis(DeviceBuffer a, int axis) {
    return reduceAxis(a, axis, false, "reduce_axis_argmax");
  }

  @Override
  public DeviceBuffer maxAxisBackward(DeviceBuffer upstream, DeviceBuffer input, int axis) {
    ensureInitialized();
    VulkanBuffer grad = vk(upstream);
    VulkanBuffer in = vk(input);
    Shape shape = in.shape();
    if (axis < 0 || axis >= shape.rank()) {
      throw new IllegalArgumentException("axis " + axis + " is out of bounds for shape " + shape);
    }
    int outer = 1;
    for (int i = 0; i < axis; i++) {
      outer = Math.multiplyExact(outer, shape.dim(i));
    }
    int inner = 1;
    for (int i = axis + 1; i < shape.rank(); i++) {
      inner = Math.multiplyExact(inner, shape.dim(i));
    }
    if (grad.count() != outer * inner) {
      throw new IllegalArgumentException("maxAxisBackward gradient shape " + grad.shape()
          + " is incompatible with input " + shape + " and axis " + axis);
    }
    VulkanBuffer out = alloc(shape);
    VulkanRuntime.dispatch("reduce_axis_max_backward", grid(outer * inner), 1, 1,
        new long[] {out.buffer, grad.buffer, in.buffer},
        new int[] {outer, shape.dim(axis), inner}, -1, 0f);
    return out;
  }

  private DeviceBuffer reduceAxis(DeviceBuffer buffer, int axis, boolean keepDims, String pipeline) {
    ensureInitialized();
    VulkanBuffer input = vk(buffer);
    Shape shape = input.shape();
    if (axis < 0 || axis >= shape.rank()) {
      throw new IllegalArgumentException("axis " + axis + " is out of bounds for shape " + shape);
    }
    int outer = 1;
    for (int i = 0; i < axis; i++) {
      outer = Math.multiplyExact(outer, shape.dim(i));
    }
    int inner = 1;
    for (int i = axis + 1; i < shape.rank(); i++) {
      inner = Math.multiplyExact(inner, shape.dim(i));
    }
    int[] outputDims;
    if (keepDims) {
      outputDims = shape.dims();
      outputDims[axis] = 1;
    } else {
      int[] dims = shape.dims();
      outputDims = new int[dims.length - 1];
      System.arraycopy(dims, 0, outputDims, 0, axis);
      System.arraycopy(dims, axis + 1, outputDims, axis, dims.length - axis - 1);
    }
    VulkanBuffer out = alloc(Shape.of(outputDims));
    VulkanRuntime.dispatch(pipeline, grid(outer * inner), 1, 1,
        new long[] {out.buffer, input.buffer},
        new int[] {outer, shape.dim(axis), inner}, -1, 0f);
    return out;
  }

  // ---- other ops -------------------------------------------------------------

  @Override
  public DeviceBuffer reluBackward(DeviceBuffer upstream, DeviceBuffer input) {
    ensureInitialized();
    VulkanBuffer grad = vk(upstream);
    VulkanBuffer in = vk(input);
    int n = grad.count();
    VulkanBuffer out = alloc(grad.shape());
    VulkanRuntime.dispatch("relu_backward", grid(n), 1, 1,
        new long[] {out.buffer, grad.buffer, in.buffer}, new int[] {n}, -1, 0f);
    return out;
  }

  @Override
  public DeviceBuffer broadcastTo(DeviceBuffer a, Shape target) {
    ensureInitialized();
    Shape src = a.shape();
    if (src.equals(target)) {
      return a;
    }
    int rank = target.rank();
    if (rank > MAX_BROADCAST_RANK) {
      throw new UnsupportedOperationException("broadcastTo supports up to rank " + MAX_BROADCAST_RANK + ", got " + target);
    }
    int pad = rank - src.rank();
    int[] srcDims = new int[rank];
    for (int i = 0; i < rank; i++) {
      srcDims[i] = i < pad ? 1 : src.dim(i - pad);
    }
    int[] srcStrides = new int[rank];
    int stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
      srcStrides[i] = srcDims[i] == 1 ? 0 : stride;
      stride *= srcDims[i];
    }
    int[] targetDims = target.dims();
    int leadPad = MAX_BROADCAST_RANK - rank;
    int[] d = new int[MAX_BROADCAST_RANK];
    int[] s = new int[MAX_BROADCAST_RANK];
    for (int i = 0; i < MAX_BROADCAST_RANK; i++) {
      d[i] = i < leadPad ? 1 : targetDims[i - leadPad];
      s[i] = i < leadPad ? 0 : srcStrides[i - leadPad];
    }
    VulkanBuffer in = vk(a);
    int n = Math.toIntExact(target.size());
    VulkanBuffer out = alloc(target);
    VulkanRuntime.dispatch("broadcast_to", grid(n), 1, 1, new long[] {out.buffer, in.buffer},
        new int[] {d[0], d[1], d[2], d[3], s[0], s[1], s[2], s[3], n}, -1, 0f);
    return out;
  }

  private static int[] convPush(int batch, ConvSpec spec) {
    return new int[] {batch, spec.inChannels(), spec.inHeight(), spec.inWidth(),
        spec.outChannels(), spec.kernel(), spec.stride(), spec.pad(),
        spec.outHeight(), spec.outWidth()};
  }

  @Override
  public DeviceBuffer conv2d(DeviceBuffer x, DeviceBuffer w, ConvSpec spec) {
    ensureInitialized();
    VulkanBuffer in = vk(x);
    VulkanBuffer filters = vk(w);
    int batch = in.shape().dim(0);
    VulkanBuffer out = alloc(spec.outputShape(batch));
    VulkanRuntime.dispatch("conv2d_fwd", grid(batch * spec.outputSize()), 1, 1,
        new long[] {out.buffer, in.buffer, filters.buffer}, convPush(batch, spec), -1, 0f);
    return out;
  }

  @Override
  public DeviceBuffer conv2dGradInput(DeviceBuffer upstream, DeviceBuffer w, ConvSpec spec) {
    ensureInitialized();
    VulkanBuffer grad = vk(upstream);
    VulkanBuffer filters = vk(w);
    int batch = grad.shape().dim(0);
    VulkanBuffer out = alloc(spec.inputShape(batch));
    VulkanRuntime.dispatch("conv2d_dx", grid(batch * spec.inputSize()), 1, 1,
        new long[] {out.buffer, grad.buffer, filters.buffer}, convPush(batch, spec), -1, 0f);
    return out;
  }

  @Override
  public DeviceBuffer conv2dGradWeight(DeviceBuffer x, DeviceBuffer upstream, ConvSpec spec) {
    ensureInitialized();
    VulkanBuffer in = vk(x);
    VulkanBuffer grad = vk(upstream);
    int batch = in.shape().dim(0);
    VulkanBuffer out = alloc(spec.weightShape());
    VulkanRuntime.dispatch("conv2d_dw", grid(spec.outChannels() * spec.weightSize()), 1, 1,
        new long[] {out.buffer, in.buffer, grad.buffer}, convPush(batch, spec), -1, 0f);
    return out;
  }

  @Override
  public void sync() {
    if (initialized) {
      VulkanRuntime.sync();
    }
  }

  @Override
  public void release(DeviceBuffer buffer) {
    vk(buffer).release();
  }

  // ---- helpers -----------------------------------------------------------

  private VulkanBuffer alloc(Shape shape) {
    long bytes = Math.max(4L, Math.multiplyExact(shape.size(), (long) DType.F32.byteSize()));
    long[] handles = VulkanRuntime.alloc(bytes);
    return new VulkanBuffer(handles[0], handles[1], shape, DType.F32, Device.vulkan());
  }

  private static int grid(int n) {
    return n == 0 ? 0 : 1 + (n - 1) / BLOCK;
  }

  private static int ceilDiv(int size, int div) {
    return (size + div - 1) / div;
  }

  private static void requireVulkanDevice(Device device) {
    if (!device.equals(Device.vulkan())) {
      throw new IllegalArgumentException("Vulkan backend currently supports only vulkan:0, got " + device);
    }
  }

  private static void requireMatchingPlacement(VulkanBuffer left, VulkanBuffer right) {
    if (left.dtype() != right.dtype() || !left.device().equals(right.device())) {
      throw new IllegalArgumentException("matmul operands must use the same device and dtype");
    }
    requireVulkanDevice(left.device());
    if (left.dtype() != DType.F32) {
      throw new UnsupportedOperationException("Vulkan matmul supports F32 only, got " + left.dtype());
    }
  }

  private static VulkanBuffer vk(DeviceBuffer buffer) {
    if (buffer instanceof VulkanBuffer vulkan) {
      return vulkan;
    }
    throw new IllegalArgumentException("expected a Vulkan buffer, got " + buffer.getClass());
  }

  private static int binaryCode(Op op) {
    return switch (op) {
      case ADD -> 0;
      case SUB -> 1;
      case MUL -> 2;
      case DIV -> 3;
      case MAX -> 4;
      case MIN -> 5;
      default -> throw new IllegalArgumentException("not a binary/scalar op: " + op);
    };
  }

  private static int unaryCode(Op op) {
    return switch (op) {
      case NEG -> 0;
      case EXP -> 1;
      case LOG -> 2;
      case SQRT -> 3;
      case RSQRT -> 4;
      case TANH -> 5;
      case SIGMOID -> 6;
      case RELU -> 7;
      case ABS -> 8;
      case SIGN -> 9;
      default -> throw new IllegalArgumentException("not a unary op: " + op);
    };
  }
}
