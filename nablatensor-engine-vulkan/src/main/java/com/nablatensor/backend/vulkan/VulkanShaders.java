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

import java.util.List;

/**
 * The GLSL compute-shader library for {@link VulkanBackend}: one {@link
 * VulkanShader} per tensor op, compiled to SPIR-V by {@code libshaderc} at
 * runtime. There is no source sharing with the CUDA / ROCm backends — those run
 * one CUDA-C translation unit ({@code GpuKernels}); this is the GLSL equivalent,
 * kept in its own file so {@link VulkanBackend} stays op-dispatch only.
 *
 * <p>Every shader is registered and dispatched by its {@link VulkanShader#name()},
 * so the name string lives here exactly once, and its storage-buffer count is
 * parsed from the source rather than hand-maintained (see {@link VulkanShader}).
 * The {@link #ALL} list is the registration order.
 */
final class VulkanShaders {

  private VulkanShaders() {
  }

  static final String COMMON = "#version 450\n";

  static final VulkanShader EW_BINARY = VulkanShader.of("ew_binary", COMMON + """
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
      """);

  static final VulkanShader EW_SCALAR = VulkanShader.of("ew_scalar", COMMON + """
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
      """);

  static final VulkanShader EW_UNARY = VulkanShader.of("ew_unary", COMMON + """
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
      """);

  static final VulkanShader TRANSPOSE = VulkanShader.of("transpose2d", COMMON + """
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
      """);

  /** matmul workgroup edge, micro-tile per invocation, and the derived block / K tiles. */
  static final int MM_WG = 16;
  static final int MM_TM = 4;
  static final int MM_TN = 4;
  static final int MM_BK = 16;
  static final int MM_BM = MM_WG * MM_TM;
  static final int MM_BN = MM_WG * MM_TN;

  /**
   * {@code o(M,N) = a(M,K) @ b(K,N)}, row-major. A {@value #MM_WG}x{@value #MM_WG}
   * workgroup; each invocation accumulates a {@value #MM_TM}x{@value #MM_TN}
   * micro-tile of the output in registers, so a {@value #MM_BM}x{@value #MM_BN}
   * block tile is computed per workgroup with {@value #MM_BK} FMAs per pair of
   * shared-memory loads. {@code As} is stored transposed so the inner-loop reads
   * are unit-stride; the cooperative loads over {@code As}/{@code Bs} are
   * {@code (BM*BK)/(WG*WG)} elements per invocation.
   *
   * <p>{@code BM}, {@code BN}, {@code BK}, {@code TM}, {@code TN}, {@code WG} are
   * substituted from the Java constants above, so the launch geometry in {@link
   * VulkanBackend#matmul} (which divides by {@link #MM_BM} / {@link #MM_BN}) and
   * the shader cannot disagree. {@code BK} / {@code BN} being powers of two lets
   * the driver strength-reduce the {@code % } / {@code / } into shifts.
   */
  private static final String MATMUL_TEMPLATE = COMMON + """
      layout(local_size_x = {WG}, local_size_y = {WG}) in;
      layout(std430, binding = 0) writeonly buffer O { float o[]; };
      layout(std430, binding = 1) readonly  buffer A { float a[]; };
      layout(std430, binding = 2) readonly  buffer B { float b[]; };
      layout(push_constant) uniform P { uint M; uint K; uint N; } p;

      const uint WG  = {WG}u;
      const uint WGS = WG * WG;
      const uint BM  = {BM}u;
      const uint BN  = {BN}u;
      const uint BK  = {BK}u;
      const uint TM  = {TM}u;
      const uint TN  = {TN}u;

      shared float As[BK * BM];   // As[k * BM + m]  (A tile, transposed into shared)
      shared float Bs[BK * BN];   // Bs[k * BN + n]

      void main() {
        uint tid  = gl_LocalInvocationID.y * WG + gl_LocalInvocationID.x;
        uint rowBase = gl_WorkGroupID.y * BM;
        uint colBase = gl_WorkGroupID.x * BN;
        uint rowT = rowBase + gl_LocalInvocationID.y * TM;
        uint colT = colBase + gl_LocalInvocationID.x * TN;

        float acc[TM][TN];
        for (uint i = 0u; i < TM; i++)
          for (uint j = 0u; j < TN; j++)
            acc[i][j] = 0.0;

        for (uint k0 = 0u; k0 < p.K; k0 += BK) {
          for (uint i = 0u; i < (BM * BK) / WGS; i++) {
            uint idx = tid + i * WGS;              // [0, BM*BK)
            uint sk = idx % BK;                    // K within tile  -> coalesced global read
            uint sm = idx / BK;                    // [0, BM) row within block
            uint gr = rowBase + sm;
            uint gk = k0 + sk;
            As[sk * BM + sm] = (gr < p.M && gk < p.K) ? a[gr * p.K + gk] : 0.0;
          }
          for (uint i = 0u; i < (BN * BK) / WGS; i++) {
            uint idx = tid + i * WGS;
            uint sn = idx % BN;                    // [0, BN) col within block -> coalesced
            uint sk = idx / BN;                    // K within tile
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

  static final VulkanShader MATMUL = VulkanShader.of("matmul", MATMUL_TEMPLATE
      .replace("{WG}", Integer.toString(MM_WG))
      .replace("{BM}", Integer.toString(MM_BM))
      .replace("{BN}", Integer.toString(MM_BN))
      .replace("{BK}", Integer.toString(MM_BK))
      .replace("{TM}", Integer.toString(MM_TM))
      .replace("{TN}", Integer.toString(MM_TN)));

  static final VulkanShader REDUCE_SUM = VulkanShader.of("reduce_sum", COMMON + """
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
      """);

  static final VulkanShader REDUCE_MAX = VulkanShader.of("reduce_max", COMMON + """
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
      """);

  static final VulkanShader SUM_AXIS0 = VulkanShader.of("sum_axis0", COMMON + """
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
      """);

  /**
   * Reduce a rank-N buffer along one axis, viewed as {@code outer × axisSize ×
   * inner}. One invocation per {@code outer*inner} output element walks the axis
   * with stride {@code inner}. Mirrors the CUDA / ROCm {@code reduce_axis_*}
   * kernels. {@code argmax} writes the winning index as a float.
   */
  private static final String REDUCE_AXIS_PUSH =
      "layout(push_constant) uniform P { uint outer; uint axisSize; uint inner; } p;\n";

  static final VulkanShader REDUCE_AXIS_SUM = VulkanShader.of("reduce_axis_sum", COMMON + """
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
      """);

  static final VulkanShader REDUCE_AXIS_MAX = VulkanShader.of("reduce_axis_max", COMMON + """
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
      """);

  static final VulkanShader REDUCE_AXIS_ARGMAX = VulkanShader.of("reduce_axis_argmax", COMMON + """
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
      """);

  static final VulkanShader REDUCE_AXIS_MAX_BACKWARD =
      VulkanShader.of("reduce_axis_max_backward", COMMON + """
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
      """);

  static final VulkanShader RELU_BACKWARD = VulkanShader.of("relu_backward", COMMON + """
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
      """);

  /**
   * Phase 1 of a two-phase reduction: a grid of workgroups, each grid-strides
   * over its share of the input and writes one partial. Phase 2 is the
   * single-workgroup {@link #REDUCE_SUM} / {@link #REDUCE_MAX} run over the
   * partials buffer. Replaces the old single-workgroup pass, which had 256
   * threads reduce the whole tensor.
   */
  static final VulkanShader REDUCE_SUM_P1 = VulkanShader.of("reduce_sum_p1", COMMON + """
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
      """);

  static final VulkanShader REDUCE_MAX_P1 = VulkanShader.of("reduce_max_p1", COMMON + """
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
      """);

  static final VulkanShader BROADCAST = VulkanShader.of("broadcast_to", COMMON + """
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
      """);

  /**
   * Direct 2-D convolution over batched, channel-major rank-2 images (see
   * {@code ConvSpec}). One output element per invocation, mirroring the CUDA /
   * ROCm {@code conv2d_fwd} / {@code conv2d_dx} / {@code conv2d_dw} kernels.
   * Push block: 10 uints {batch,inC,inH,inW,outC,k,stride,pad,outH,outW}.
   */
  private static final String CONV_PUSH =
      "layout(push_constant) uniform P { uint batch, inC, inH, inW, outC, k, stride, pad, outH, outW; } p;\n";

  static final VulkanShader CONV2D_FWD = VulkanShader.of("conv2d_fwd", COMMON + """
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
      """);

  static final VulkanShader CONV2D_DX = VulkanShader.of("conv2d_dx", COMMON + """
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
      """);

  static final VulkanShader CONV2D_DW = VulkanShader.of("conv2d_dw", COMMON + """
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
      """);

  /** Every static shader, in registration / module-load order. */
  static final List<VulkanShader> ALL = List.of(
      EW_BINARY, EW_SCALAR, EW_UNARY, TRANSPOSE, MATMUL,
      REDUCE_SUM, REDUCE_MAX, REDUCE_SUM_P1, REDUCE_MAX_P1, SUM_AXIS0,
      REDUCE_AXIS_SUM, REDUCE_AXIS_MAX, REDUCE_AXIS_ARGMAX, REDUCE_AXIS_MAX_BACKWARD,
      RELU_BACKWARD, BROADCAST, CONV2D_FWD, CONV2D_DX, CONV2D_DW);
}
