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

import java.util.function.Consumer;

/**
 * Records one pass of a valuation into an {@link AadTape}.
 *
 * <pre>{@code
 * AadTape tape = AadRecorder.record(rec -> {
 *   SDouble s0 = rec.input("S0", 100.0);
 *   SDouble s = s0;
 *   for (int t = 0; t < steps; t++) {
 *     s = s.mul(rec.randn().mul(vol).add(drift).exp());
 *   }
 *   rec.output(s.sub(strike).max(0.0));
 * });
 * }</pre>
 */
public final class AadRecorder {

  private final AadTape.Builder builder = new AadTape.Builder();

  private AadRecorder() {
  }

  public static AadTape record(Consumer<AadRecorder> body) {
    AadRecorder recorder = new AadRecorder();
    body.accept(recorder);
    return recorder.builder.build();
  }

  SDouble node(AadOp op, int a, int b) {
    return new SDouble(this, builder.add(op, a, b, 0.0));
  }

  /** A differentiable input; its recorded value is the default for a replay. */
  public SDouble input(String name, double value) {
    return new SDouble(this, builder.addInput(name, value));
  }

  public SDouble constant(double value) {
    return new SDouble(this, builder.add(AadOp.CONST, -1, -1, value));
  }

  /**
   * A standard normal that varies per scenario, drawn from the default stream.
   * Nothing is drawn while recording: the node only reserves the next slot in
   * the per-scenario random stream, which the replay kernel generates on-device.
   */
  public SDouble randn() {
    return new SDouble(this, builder.addRandn());
  }

  /** A uniform draw on {@code [0, 1)} that varies per scenario, from the default stream. */
  public SDouble randu() {
    return new SDouble(this, builder.addRandu("default"));
  }

  /**
   * An independent named random stream. Draws from a stream are statistically
   * independent of every other stream and of the default one, so a jump-time
   * clock, a low-discrepancy dimension or a second factor can be advanced
   * without perturbing the primary path. {@code stream("default")} is the same
   * stream {@link #randn()} / {@link #randu()} use.
   */
  public RandomStream stream(String name) {
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("stream name must be non-blank");
    }
    return new RandomStream(name);
  }

  /** A handle to one named {@linkplain #stream(String) random stream}. */
  public final class RandomStream {

    private final String name;

    private RandomStream(String name) {
      this.name = name;
    }

    /** A standard normal from this stream. */
    public SDouble randn() {
      return new SDouble(AadRecorder.this, builder.addRandn(name));
    }

    /** A uniform draw on {@code [0, 1)} from this stream. */
    public SDouble randu() {
      return new SDouble(AadRecorder.this, builder.addRandu(name));
    }
  }

  /** Records the single output of this valuation, named {@code "value"}. */
  public void output(SDouble value) {
    builder.setOutput(value.node);
  }

  /**
   * Records one of several named outputs. Every reverse sweep over the tape then
   * yields this output's value, its Monte-Carlo standard error and its full
   * input gradient; the names address the rows of {@link AadResult}. A name may
   * be recorded only once, and mixing this with {@link #output(SDouble)} (which
   * uses the name {@code "value"}) is allowed as long as the names stay distinct.
   */
  public void output(String name, SDouble value) {
    builder.addOutput(name, value.node);
  }
}
