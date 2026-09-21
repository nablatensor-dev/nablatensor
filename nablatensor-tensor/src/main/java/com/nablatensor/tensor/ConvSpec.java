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
package com.nablatensor.tensor;

import com.nablatensor.codegen.Of;

/**
 * The geometry of a 2-D convolution over batched, channel-major images.
 *
 * <p>Everything stays rank-2, which is the shape the rest of the library
 * already speaks: one image per row, channels laid out one after another.
 *
 * <pre>
 *   input    (batch, inChannels * inHeight * inWidth)
 *   weights  (outChannels, inChannels * kernel * kernel)
 *   output   (batch, outChannels * outHeight * outWidth)
 * </pre>
 *
 * <p>Square kernels only, and no bias: a bias is one broadcast add away and
 * does not need to be inside the convolution.
 */
@Of
public final class ConvSpec {

  private final int inChannels;
  private final int inHeight;
  private final int inWidth;
  private final int outChannels;
  private final int kernel;
  private final int stride;
  private final int pad;
  static ConvSpec create(int inChannels, int inHeight, int inWidth, int outChannels, int kernel, int stride, int pad) {
    return new ConvSpec(inChannels, inHeight, inWidth, outChannels, kernel, stride, pad);
  }

  public static ConvSpecBuilder of() {
    return new ConvSpecBuilder();
  }

  public int inChannels() {
    return inChannels;
  }

  public int inHeight() {
    return inHeight;
  }

  public int inWidth() {
    return inWidth;
  }

  public int outChannels() {
    return outChannels;
  }

  public int kernel() {
    return kernel;
  }

  public int stride() {
    return stride;
  }

  public int pad() {
    return pad;
  }


  private ConvSpec(int inChannels, int inHeight, int inWidth, int outChannels, int kernel, int stride, int pad) {
    if (inChannels < 1 || inHeight < 1 || inWidth < 1 || outChannels < 1) {
      throw new IllegalArgumentException("channels and spatial dims must be positive");
    }
    if (kernel < 1 || stride < 1 || pad < 0) {
      throw new IllegalArgumentException("kernel/stride must be positive, pad non-negative");
    }
    if (inHeight + 2 * pad < kernel || inWidth + 2 * pad < kernel) {
      throw new IllegalArgumentException("kernel " + kernel + " does not fit "
          + inHeight + "x" + inWidth + " with pad " + pad);
    }
    this.inChannels = inChannels;
    this.inHeight = inHeight;
    this.inWidth = inWidth;
    this.outChannels = outChannels;
    this.kernel = kernel;
    this.stride = stride;
    this.pad = pad;
  }

  public int outHeight() {
    return (inHeight + 2 * pad - kernel) / stride + 1;
  }

  public int outWidth() {
    return (inWidth + 2 * pad - kernel) / stride + 1;
  }

  /** Elements per input image: {@code inChannels * inHeight * inWidth}. */
  public int inputSize() {
    return inChannels * inHeight * inWidth;
  }

  /** Elements per output image: {@code outChannels * outHeight * outWidth}. */
  public int outputSize() {
    return outChannels * outHeight() * outWidth();
  }

  /** Elements per filter: {@code inChannels * kernel * kernel}. */
  public int weightSize() {
    return inChannels * kernel * kernel;
  }

  public Shape weightShape() {
    return Shape.of(outChannels, weightSize());
  }

  public Shape inputShape(int batch) {
    return Shape.of(batch, inputSize());
  }

  public Shape outputShape(int batch) {
    return Shape.of(batch, outputSize());
  }

  /** The spec of a layer that consumes this layer's output. */
  public ConvSpec next(int nextOutChannels, int nextKernel, int nextStride, int nextPad) {
    return ConvSpec.of().inChannels(outChannels).inHeight(outHeight()).inWidth(outWidth()).outChannels(nextOutChannels).kernel(nextKernel).stride(nextStride).pad(nextPad).build();
  }

  int batchOf(Shape shape, String what) {
    if (shape.rank() != 2) {
      throw new IllegalArgumentException(what + " must be rank-2, got " + shape);
    }
    return shape.dim(0);
  }
}
