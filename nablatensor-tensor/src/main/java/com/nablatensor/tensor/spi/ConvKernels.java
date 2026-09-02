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

import java.util.List;

/**
 * Direct (non-im2col) square-kernel 2-D convolution and its two gradients. All
 * three run one thread per element of their own output, so none of them needs an
 * atomic: forward owns an output pixel, {@code conv2d_dx} owns an input pixel,
 * {@code conv2d_dw} owns a weight.
 */
final class ConvKernels {

  private ConvKernels() {
  }

  static final GpuKernel CONV2D_FWD = GpuKernel.of("""
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
      """);

  /** The {@code % stride != 0} guards skip taps that no output pixel reads under striding. */
  static final GpuKernel CONV2D_DX = GpuKernel.of("""
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
      """);

  static final GpuKernel CONV2D_DW = GpuKernel.of("""
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
      """);

  static final List<GpuKernel> KERNELS = List.of(CONV2D_FWD, CONV2D_DX, CONV2D_DW);
}
