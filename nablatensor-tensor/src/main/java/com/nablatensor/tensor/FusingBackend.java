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

import com.nablatensor.tensor.expr.Expr;
import com.nablatensor.tensor.spi.ComputeBackend;
import com.nablatensor.tensor.spi.DeviceBuffer;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Wraps a real backend during {@link Jit} tracing. Elementwise ops
 * (add/sub/mul/div/unary/scalar) build an {@link Expr} graph instead of
 * running eagerly; matmul/transpose/download are fusion boundaries that
 * materialize a real buffer via {@code delegate.fused(...)} first.
 */
final class FusingBackend implements ComputeBackend {

  private final ComputeBackend delegate;
  private final List<DeviceBuffer> owned = new ArrayList<>();

  FusingBackend(ComputeBackend delegate) {
    this.delegate = delegate;
  }

  @Override
  public String name() {
    return delegate.name();
  }

  @Override
  public DeviceType deviceType() {
    return delegate.deviceType();
  }

  @Override
  public boolean isAvailable() {
    return delegate.isAvailable();
  }

  @Override
  public int priority() {
    return delegate.priority();
  }

  /** Wraps an already-real buffer as a trivial (unfused) traced leaf. */
  ExprBuffer leaf(DeviceBuffer real) {
    return new ExprBuffer(new Expr.Input(0), List.of(real), real.shape(), real.dtype(), real.device());
  }

  @Override
  public DeviceBuffer upload(float[] data, Shape shape, DType dtype, Device device) {
    return leaf(delegate.upload(data, shape, dtype, device));
  }

  @Override
  public float[] download(DeviceBuffer buffer) {
    return delegate.download(materialize(buffer));
  }

  @Override
  public DeviceBuffer binary(Op op, DeviceBuffer a, DeviceBuffer b) {
    ExprBuffer left = traced(a);
    ExprBuffer right = traced(b);
    Merged merged = merge(left, right);
    return new ExprBuffer(new Expr.Binary(op, merged.left, merged.right), merged.leaves,
        left.shape(), left.dtype(), left.device());
  }

  @Override
  public DeviceBuffer scalar(Op op, DeviceBuffer a, double value) {
    ExprBuffer in = traced(a);
    return new ExprBuffer(new Expr.Scalar(op, in.expr(), value), in.leaves(), in.shape(), in.dtype(), in.device());
  }

  @Override
  public DeviceBuffer unary(Op op, DeviceBuffer a) {
    ExprBuffer in = traced(a);
    return new ExprBuffer(new Expr.Unary(op, in.expr()), in.leaves(), in.shape(), in.dtype(), in.device());
  }

  @Override
  public DeviceBuffer transpose(DeviceBuffer a) {
    DeviceBuffer result = delegate.transpose(materialize(a));
    owned.add(result);
    return leaf(result);
  }

  @Override
  public DeviceBuffer matmul(DeviceBuffer a, DeviceBuffer b) {
    DeviceBuffer result = delegate.matmul(materialize(a), materialize(b));
    owned.add(result);
    return leaf(result);
  }

  @Override
  public DeviceBuffer batchedMatmul(DeviceBuffer a, DeviceBuffer b) {
    DeviceBuffer result = delegate.batchedMatmul(materialize(a), materialize(b));
    owned.add(result);
    return leaf(result);
  }

  @Override
  public DeviceBuffer sliceAxis0(DeviceBuffer input, int index) {
    DeviceBuffer result = delegate.sliceAxis0(materialize(input), index);
    owned.add(result);
    return leaf(result);
  }

  @Override
  public DeviceBuffer stackAxis0(DeviceBuffer[] inputs) {
    DeviceBuffer[] materialized = new DeviceBuffer[inputs.length];
    for (int i = 0; i < inputs.length; i++) {
      materialized[i] = materialize(inputs[i]);
    }
    DeviceBuffer result = delegate.stackAxis0(materialized);
    owned.add(result);
    return leaf(result);
  }

  @Override
  public DeviceBuffer reduceSum(DeviceBuffer a) {
    DeviceBuffer result = delegate.reduceSum(materialize(a));
    owned.add(result);
    return leaf(result);
  }

  @Override
  public DeviceBuffer reduceMax(DeviceBuffer a) {
    DeviceBuffer result = delegate.reduceMax(materialize(a));
    owned.add(result);
    return leaf(result);
  }

  @Override
  public DeviceBuffer conv2d(DeviceBuffer x, DeviceBuffer w, ConvSpec spec) {
    DeviceBuffer result = delegate.conv2d(x, w, spec);
    owned.add(result);
    return result;
  }

  @Override
  public DeviceBuffer conv2dGradInput(DeviceBuffer upstream, DeviceBuffer w, ConvSpec spec) {
    DeviceBuffer result = delegate.conv2dGradInput(upstream, w, spec);
    owned.add(result);
    return result;
  }

  @Override
  public DeviceBuffer conv2dGradWeight(DeviceBuffer x, DeviceBuffer upstream, ConvSpec spec) {
    DeviceBuffer result = delegate.conv2dGradWeight(x, upstream, spec);
    owned.add(result);
    return result;
  }

  @Override
  public DeviceBuffer reluBackward(DeviceBuffer upstream, DeviceBuffer input) {
    DeviceBuffer result = delegate.reluBackward(upstream, input);
    owned.add(result);
    return result;
  }

  @Override
  public DeviceBuffer sumAxis0(DeviceBuffer a) {
    DeviceBuffer result = delegate.sumAxis0(materialize(a));
    owned.add(result);
    return leaf(result);
  }

  @Override
  public DeviceBuffer broadcastTo(DeviceBuffer a, Shape target) {
    DeviceBuffer result = delegate.broadcastTo(materialize(a), target);
    owned.add(result);
    return leaf(result);
  }

  @Override
  public void sync() {
    delegate.sync();
  }

  /** Forces a lazily-fused chain into a real backend buffer, running it as one kernel. */
  DeviceBuffer materialize(DeviceBuffer buffer) {
    ExprBuffer traced = traced(buffer);
    if (traced.expr() instanceof Expr.Input && traced.leaves().size() == 1) {
      return traced.leaves().get(0);
    }
    DeviceBuffer result = delegate.fused(traced.expr(), traced.leaves().toArray(DeviceBuffer[]::new));
    owned.add(result);
    return result;
  }

  /** Releases every hard-break intermediate this trace allocated, except {@code keep} (the final result). */
  void releaseIntermediates(DeviceBuffer keep) {
    for (DeviceBuffer buffer : owned) {
      if (buffer != keep) {
        delegate.release(buffer);
      }
    }
  }

  private static ExprBuffer traced(DeviceBuffer buffer) {
    if (buffer instanceof ExprBuffer expr) {
      return expr;
    }
    throw new IllegalStateException("expected a traced buffer, got " + buffer.getClass());
  }

  private record Merged(Expr left, Expr right, List<DeviceBuffer> leaves) {
  }

  /** Combines two chains' leaf lists (deduped by reference) and rewrites the right side's indices. */
  private static Merged merge(ExprBuffer left, ExprBuffer right) {
    Map<DeviceBuffer, Integer> index = new IdentityHashMap<>();
    List<DeviceBuffer> leaves = new ArrayList<>();
    for (DeviceBuffer buffer : left.leaves()) {
      index.computeIfAbsent(buffer, b -> add(leaves, b));
    }
    Expr rewrittenRight = reindex(right.expr(), right.leaves(), index, leaves);
    return new Merged(left.expr(), rewrittenRight, List.copyOf(leaves));
  }

  private static Expr reindex(Expr expr, List<DeviceBuffer> originalLeaves,
      Map<DeviceBuffer, Integer> index, List<DeviceBuffer> combined) {
    return switch (expr) {
      case Expr.Input in -> {
        DeviceBuffer buffer = originalLeaves.get(in.index());
        yield new Expr.Input(index.computeIfAbsent(buffer, b -> add(combined, b)));
      }
      case Expr.Unary u -> new Expr.Unary(u.op(), reindex(u.in(), originalLeaves, index, combined));
      case Expr.Binary b -> new Expr.Binary(b.op(),
          reindex(b.left(), originalLeaves, index, combined),
          reindex(b.right(), originalLeaves, index, combined));
      case Expr.Scalar s -> new Expr.Scalar(s.op(), reindex(s.in(), originalLeaves, index, combined), s.value());
    };
  }

  private static int add(List<DeviceBuffer> leaves, DeviceBuffer buffer) {
    leaves.add(buffer);
    return leaves.size() - 1;
  }
}
