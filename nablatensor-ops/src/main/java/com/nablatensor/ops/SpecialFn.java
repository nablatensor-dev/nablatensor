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
 * Tape-level special functions the primitive op set does not provide directly:
 * the normal CDF/PDF and a general power.
 *
 * <p>{@link #normCdf} uses the Page/Bagby logistic approximation
 * {@code 1 / (1 + exp(-(1.5976 x + 0.070566 x^3)))}: everywhere smooth (so its
 * adjoint is well behaved), odd about the mean, and accurate to about
 * {@code 1.4e-4} absolute. For a reference-grade value use
 * {@code com.nablatensor.quant.BlackScholes.N} — that one is not tape-safe.
 */
public final class SpecialFn {

  private static final double SQRT2 = Math.sqrt(2.0);
  private static final double INV_SQRT_2PI = 1.0 / Math.sqrt(2.0 * Math.PI);

  private SpecialFn() {
  }

  /** Standard normal CDF, {@code N(x)}. Smooth logistic approximation. */
  public static SDouble normCdf(AadRecorder rec, SDouble x) {
    SDouble poly = x.mul(1.5976).add(x.mul(x).mul(x).mul(0.070566)); // 1.5976 x + 0.070566 x^3
    return rec.constant(1.0).div(poly.neg().exp().add(1.0));
  }

  /** Standard normal PDF, {@code phi(x)}. Exact. */
  public static SDouble normPdf(SDouble x) {
    return x.mul(x).mul(-0.5).exp().mul(INV_SQRT_2PI);
  }

  /** {@code erf(x)}, via {@code erf(x) = 2 N(x sqrt 2) - 1}. */
  public static SDouble erf(AadRecorder rec, SDouble x) {
    return normCdf(rec, x.mul(SQRT2)).mul(2.0).sub(1.0);
  }

  /** {@code x ^ p} for a constant exponent; requires {@code x > 0}. */
  public static SDouble pow(SDouble x, double p) {
    if (p == 1.0) {
      return x;
    }
    if (p == 0.0) {
      return x.div(x); // 1, but keeps x on the tape so the node count is stable
    }
    return x.log().mul(p).exp();
  }

  /** {@code x ^ p} for a differentiable exponent; requires {@code x > 0}. */
  public static SDouble pow(SDouble x, SDouble p) {
    return x.log().mul(p).exp();
  }
}
