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
package com.nablatensor.ops;

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.SDouble;

/**
 * Smoothed indicators: differentiable stand-ins for the discontinuous
 * {@code STEP} / {@code GT} / {@code 1{a<x<b}} that path-dependent payoffs need
 * for barrier monitoring and digital settlement.
 *
 * <p>Each takes a {@code width} in the same units as its argument. The functions
 * are built entirely from the engine's primitive ops, so their adjoint is the
 * exact derivative of the smoothed form (no special-cased reverse rule) and they
 * run unchanged on every backend. As {@code width -> 0} the smoothed value
 * converges to the discontinuous limit; its derivative concentrates into a
 * spike, so a barrier delta computed this way is a genuine (mollified) delta,
 * not a bumped one.
 *
 * <p>Rule of thumb: {@code width} a fraction of a percent of the monitored
 * quantity's scale (e.g. {@code 0.01 * spot}). Too small re-introduces the
 * variance a discontinuity brings; too large biases the price.
 */
public final class Smooth {

  private Smooth() {
  }

  /** Logistic step: {@code ~1} for {@code x >> 0}, {@code ~0} for {@code x << 0}, {@code 0.5} at 0. */
  public static SDouble step(AadRecorder rec, SDouble x, double width) {
    requireWidth(width);
    SDouble e = x.div(width).neg().exp();          // exp(-x/width)
    return rec.constant(1.0).div(e.add(1.0));      // 1 / (1 + exp(-x/width))
  }

  /** Smoothed {@code 1{a > b}}. */
  public static SDouble gt(AadRecorder rec, SDouble a, SDouble b, double width) {
    return step(rec, a.sub(b), width);
  }

  /** Smoothed {@code 1{a > level}}. */
  public static SDouble gt(AadRecorder rec, SDouble a, double level, double width) {
    return step(rec, a.sub(level), width);
  }

  /** Smoothed {@code 1{a < level}}. */
  public static SDouble lt(AadRecorder rec, SDouble a, double level, double width) {
    return step(rec, a.neg().add(level), width);
  }

  /** Smoothed {@code 1{a < b}}. */
  public static SDouble lt(AadRecorder rec, SDouble a, SDouble b, double width) {
    return step(rec, b.sub(a), width);
  }

  /** Smoothed band indicator {@code 1{lo < x < hi}}. */
  public static SDouble between(AadRecorder rec, SDouble x, double lo, double hi, double width) {
    return gt(rec, x, lo, width).mul(lt(rec, x, hi, width));
  }

  /**
   * Numerically stable softplus: a smoothed {@code max(x, 0)}.
   * {@code width -> 0} recovers the hockey-stick; useful for a smoothed option
   * intrinsic when the whole payoff must stay differentiable.
   */
  public static SDouble ramp(AadRecorder rec, SDouble x, double width) {
    requireWidth(width);
    SDouble soft = x.abs().div(width).neg().exp().add(1.0).log().mul(width); // width*log(1+exp(-|x|/width))
    return x.max(0.0).add(soft);
  }

  private static void requireWidth(double width) {
    if (!(width > 0.0)) {
      throw new IllegalArgumentException("smoothing width must be > 0, got " + width);
    }
  }
}
