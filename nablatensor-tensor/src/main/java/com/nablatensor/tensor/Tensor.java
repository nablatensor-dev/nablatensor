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

import com.nablatensor.tensor.spi.ComputeBackend;
import com.nablatensor.tensor.spi.DeviceBuffer;

import java.util.List;

/**
 * An immutable, fluent tensor handle. Every operation returns a new
 * {@code Tensor}; materialization happens on the backend as operations are
 * issued, and {@link #eval()} / {@link #blockUntilReady()} wait for completion.
 *
 * <p>Backends normally reclaim device memory only once a wrapper becomes
 * unreachable and the GC/Cleaner get around to it, which is heuristic and can
 * lag under sustained allocation pressure. Tight loops that discard a result
 * every iteration (benchmarks, training steps) should call {@link #close()}
 * (or use try-with-resources) to free it deterministically instead.
 */
public final class Tensor implements AutoCloseable {

  private final ComputeBackend backend;
  private final DeviceBuffer buffer;

  Tensor(ComputeBackend backend, DeviceBuffer buffer) {
    this.backend = backend;
    this.buffer = buffer;
  }

  // ---- metadata -----------------------------------------------------------

  public Shape shape() {
    return buffer.shape();
  }

  public DType dtype() {
    return buffer.dtype();
  }

  public Device device() {
    return buffer.device();
  }

  DeviceBuffer buffer() {
    return buffer;
  }

  ComputeBackend backend() {
    return backend;
  }

  // ---- device / placement -------------------------------------------------

  /** Move to another device, re-uploading through the target backend. */
  public Tensor on(Device device) {
    if (device.equals(device())) {
      return this;
    }
    ComputeBackend target = BackendRegistry.forDevice(device);
    DeviceBuffer moved = target.upload(backend.download(buffer), shape(), dtype(), device);
    return new Tensor(target, moved);
  }

  public Tensor f32() {
    return this; // single supported floating dtype in phase 0
  }

  public Tensor dtype(DType dtype) {
    if (dtype != DType.F32) {
      throw new UnsupportedOperationException("phase 0 supports F32 only; requested " + dtype);
    }
    return this;
  }

  // ---- binary elementwise -------------------------------------------------

  public Tensor add(Tensor other) {
    return binaryBroadcast(Op.ADD, other);
  }

  public Tensor sub(Tensor other) {
    return binaryBroadcast(Op.SUB, other);
  }

  public Tensor mul(Tensor other) {
    return binaryBroadcast(Op.MUL, other);
  }

  public Tensor div(Tensor other) {
    return binaryBroadcast(Op.DIV, other);
  }

  public Tensor maximum(Tensor other) {
    return binaryBroadcast(Op.MAX, other);
  }

  public Tensor minimum(Tensor other) {
    return binaryBroadcast(Op.MIN, other);
  }

  /**
   * Binary elementwise op with NumPy-style broadcasting: operands with
   * differing shapes are each expanded to their common {@link Shape#broadcast}
   * shape first. Any tensor materialized just for this broadcast is closed
   * immediately after the op runs, so no extra buffers linger.
   */
  private Tensor binaryBroadcast(Op op, Tensor other) {
    Shape ls = shape();
    Shape rs = other.shape();
    if (ls.equals(rs)) {
      return wrap(backend.binary(op, buffer, other.buffer));
    }
    Shape target = Shape.broadcast(ls, rs);
    Tensor lb = ls.equals(target) ? null : wrap(backend.broadcastTo(buffer, target));
    Tensor rb = rs.equals(target) ? null : wrap(backend.broadcastTo(other.buffer, target));
    try {
      DeviceBuffer lBuf = lb != null ? lb.buffer : buffer;
      DeviceBuffer rBuf = rb != null ? rb.buffer : other.buffer;
      return wrap(backend.binary(op, lBuf, rBuf));
    } finally {
      if (lb != null) {
        lb.close();
      }
      if (rb != null) {
        rb.close();
      }
    }
  }

  // ---- scalar elementwise -------------------------------------------------

  public Tensor add(double value) {
    return wrap(backend.scalar(Op.ADD, buffer, value));
  }

  public Tensor sub(double value) {
    return wrap(backend.scalar(Op.SUB, buffer, value));
  }

  public Tensor mul(double value) {
    return wrap(backend.scalar(Op.MUL, buffer, value));
  }

  public Tensor div(double value) {
    return wrap(backend.scalar(Op.DIV, buffer, value));
  }

  public Tensor maximum(double value) {
    return wrap(backend.scalar(Op.MAX, buffer, value));
  }

  public Tensor minimum(double value) {
    return wrap(backend.scalar(Op.MIN, buffer, value));
  }

  // ---- unary elementwise --------------------------------------------------

  public Tensor neg() {
    return wrap(backend.unary(Op.NEG, buffer));
  }

  public Tensor exp() {
    return wrap(backend.unary(Op.EXP, buffer));
  }

  public Tensor log() {
    return wrap(backend.unary(Op.LOG, buffer));
  }

  public Tensor sqrt() {
    return wrap(backend.unary(Op.SQRT, buffer));
  }

  public Tensor rsqrt() {
    return wrap(backend.unary(Op.RSQRT, buffer));
  }

  public Tensor tanh() {
    return wrap(backend.unary(Op.TANH, buffer));
  }

  public Tensor sigmoid() {
    return wrap(backend.unary(Op.SIGMOID, buffer));
  }

  public Tensor relu() {
    return wrap(backend.unary(Op.RELU, buffer));
  }

  public Tensor abs() {
    return wrap(backend.unary(Op.ABS, buffer));
  }

  public Tensor sign() {
    return wrap(backend.unary(Op.SIGN, buffer));
  }

  // ---- movement / contraction --------------------------------------------

  public Tensor transpose() {
    return wrap(backend.transpose(buffer));
  }

  public Tensor matmul(Tensor other) {
    return wrap(backend.matmul(buffer, other.buffer));
  }

  // ---- factorizations ---------------------------------------------------

  /** Cholesky factor {@code L} of this SPD matrix ({@code this = L·Lᵀ}). See {@link Linalg#cholesky}. */
  public Tensor cholesky() {
    return Linalg.cholesky(this);
  }

  /** LU factorization with partial pivoting. See {@link Linalg#lu}. */
  public Linalg.Lu lu() {
    return Linalg.lu(this);
  }

  /** Householder QR factorization. See {@link Linalg#qr}. */
  public Linalg.Qr qr() {
    return Linalg.qr(this);
  }

  /** Symmetric eigendecomposition (cyclic Jacobi). See {@link Linalg#eigh}. */
  public Linalg.Eigh eigh() {
    return Linalg.eigh(this);
  }

  /** Thin singular value decomposition (one-sided Jacobi). See {@link Linalg#svd}. */
  public Linalg.Svd svd() {
    return Linalg.svd(this);
  }

  /** Solves {@code this · X = b} for {@code X}. See {@link Linalg#solve}. */
  public Tensor solve(Tensor b) {
    return Linalg.solve(this, b);
  }

  /** The inverse of this square matrix. See {@link Linalg#inv}. */
  public Tensor inv() {
    return Linalg.inv(this);
  }

  /** The determinant of this square matrix. See {@link Linalg#det}. */
  public double det() {
    return Linalg.det(this);
  }

  /** Batched matrix multiplication: {@code (B,M,K) x (B,K,N) -> (B,M,N)}. */
  public Tensor batchedMatmul(Tensor other) {
    if (backend != other.backend
        || !device().equals(other.device())
        || dtype() != other.dtype()) {
      throw new IllegalArgumentException(
          "batchedMatmul operands must use the same backend, device, and dtype");
    }
    return wrap(backend.batchedMatmul(buffer, other.buffer));
  }

  /** Copies one leading-axis slice and removes the leading dimension. */
  public Tensor sliceAxis0(int index) {
    return wrap(backend.sliceAxis0(buffer, index));
  }

  /** Stacks equally-shaped tensors along a new leading axis. */
  public static Tensor stackAxis0(List<Tensor> tensors) {
    if (tensors.isEmpty()) {
      throw new IllegalArgumentException("stackAxis0 requires at least one tensor");
    }
    Tensor first = tensors.getFirst();
    DeviceBuffer[] buffers = new DeviceBuffer[tensors.size()];
    for (int i = 0; i < tensors.size(); i++) {
      Tensor tensor = tensors.get(i);
      if (tensor.backend != first.backend) {
        throw new IllegalArgumentException("stackAxis0 tensors must use the same backend");
      }
      buffers[i] = tensor.buffer;
    }
    return new Tensor(first.backend, first.backend.stackAxis0(buffers));
  }

  /** Returns an independent tensor with the same elements and a different shape. */
  public Tensor reshape(int... dims) {
    Shape target = shape().reshape(dims);
    return wrap(backend.reshape(buffer, target));
  }

  // ---- convolution --------------------------------------------------------

  /**
   * 2-D convolution of batched images (one image per row) against
   * {@code weights} of shape {@code (outChannels, inChannels * k * k)}.
   */
  public Tensor conv2d(Tensor weights, ConvSpec spec) {
    return wrap(backend.conv2d(buffer, weights.buffer, spec));
  }

  /** Gradient of {@link #conv2d} with respect to the input it was given. */
  public Tensor conv2dGradInput(Tensor weights, ConvSpec spec) {
    return wrap(backend.conv2dGradInput(buffer, weights.buffer, spec));
  }

  /** Gradient of {@link #conv2d} with respect to its weights; {@code this} is the input. */
  public Tensor conv2dGradWeight(Tensor upstream, ConvSpec spec) {
    return wrap(backend.conv2dGradWeight(buffer, upstream.buffer, spec));
  }

  /** Routes {@code this} (an upstream gradient) through a ReLU evaluated at {@code input}. */
  public Tensor reluBackward(Tensor input) {
    return wrap(backend.reluBackward(buffer, input.buffer));
  }

  // ---- reductions ---------------------------------------------------------

  /** Sums every element, returning a single-element (shape {@code (1)}) tensor. */
  public Tensor sum() {
    return wrap(backend.reduceSum(buffer));
  }

  /** Mean over every element, returning a single-element (shape {@code (1)}) tensor. */
  public Tensor mean() {
    try (Tensor total = sum()) {
      return total.div((double) shape().size());
    }
  }

  /** Maximum over every element, returning a single-element (shape {@code (1)}) tensor. */
  public Tensor max() {
    return wrap(backend.reduceMax(buffer));
  }

  /** Sums a rank-2 tensor over axis 0 (the batch axis), returning shape {@code (cols)}. */
  public Tensor sumAxis0() {
    if (shape().rank() != 2) {
      throw new IllegalStateException("sumAxis0 requires a rank-2 tensor, got " + shape());
    }
    return wrap(backend.sumAxis0(buffer));
  }

  public Tensor sum(int axis) {
    return sum(axis, false);
  }

  public Tensor sum(int axis, boolean keepDims) {
    return wrap(backend.sumAxis(buffer, normalizeAxis(axis), keepDims));
  }

  public Tensor max(int axis) {
    return max(axis, false);
  }

  public Tensor max(int axis, boolean keepDims) {
    return wrap(backend.maxAxis(buffer, normalizeAxis(axis), keepDims));
  }

  /**
   * Returns maximum-value indices along {@code axis} as F32 values. Indices up to 2^24 are exact.
   */
  public Tensor argmax(int axis) {
    return wrap(backend.argMaxAxis(buffer, normalizeAxis(axis)));
  }

  /** Routes reduced gradients to the first maximum position along {@code axis}. */
  public Tensor maxAxisBackward(Tensor upstream, int axis) {
    return wrap(backend.maxAxisBackward(upstream.buffer, buffer, normalizeAxis(axis)));
  }

  /** Broadcasts to {@code target}, following {@link Shape#broadcast}'s rules. */
  public Tensor broadcastTo(Shape target) {
    return shape().equals(target) ? this : wrap(backend.broadcastTo(buffer, target));
  }

  private int normalizeAxis(int axis) {
    int rank = shape().rank();
    int normalized = axis < 0 ? axis + rank : axis;
    if (normalized < 0 || normalized >= rank) {
      throw new IllegalArgumentException("axis " + axis + " is out of bounds for shape " + shape());
    }
    return normalized;
  }

  // ---- terminals ----------------------------------------------------------

  /** Force completion of all pending kernels and return this handle. */
  public Tensor eval() {
    backend.sync();
    return this;
  }

  public Tensor blockUntilReady() {
    return eval();
  }

  /**
   * Deterministically frees this tensor's device buffer now, instead of
   * waiting on GC/Cleaner timing. Safe to call even if the GC later also
   * reclaims the same (by-then-unreachable) buffer - backends guarantee the
   * underlying free only ever runs once. Do not use this tensor after closing it.
   */
  @Override
  public void close() {
    backend.release(buffer);
  }

  public float[] toFloatArray() {
    return backend.download(buffer);
  }

  public float[][] toFloat2D() {
    Shape shape = shape();
    if (shape.rank() != 2) {
      throw new IllegalStateException("toFloat2D requires a rank-2 tensor, got " + shape);
    }
    int rows = shape.dim(0);
    int cols = shape.dim(1);
    float[] flat = toFloatArray();
    float[][] out = new float[rows][cols];
    for (int r = 0; r < rows; r++) {
      System.arraycopy(flat, r * cols, out[r], 0, cols);
    }
    return out;
  }

  public float item() {
    float[] data = toFloatArray();
    if (data.length != 1) {
      throw new IllegalStateException("item() requires a single-element tensor, got " + data.length);
    }
    return data[0];
  }

  private Tensor wrap(DeviceBuffer result) {
    return new Tensor(backend, result);
  }

  @Override
  public String toString() {
    return "Tensor(shape=" + shape() + ", dtype=" + dtype() + ", device=" + device() + ")";
  }
}
