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
package com.nablatensor.engine.jit;

/**
 * Reduced-accuracy scalar transcendentals for the Box-Muller step, used only
 * when {@code -Dnablatensor.jit.trig=fast} / {@code -Dnablatensor.jit.log=fast}
 * are set. They exploit the fact that the inputs are already range-bounded
 * ({@code u1 in (0,1]}, angle in {@code [0,2*pi)}), so the full argument
 * reduction that {@link java.lang.Math}'s intrinsics pay for is unnecessary.
 *
 * <p>Trading it changes the sample stream: prices then match the reference
 * engines to about four significant figures rather than bit for bit, the same
 * footing the SIMD engine is on.
 */
final class JitFastMath {

  private JitFastMath() {
  }

  private static final double LN2 = 0.6931471805599453;
  private static final double TWO_PI = 6.283185307179586;
  private static final double HALF_PI = 1.5707963267948966;

  /**
   * {@code log(x)} for {@code x in (0, 2]}: split off the binary exponent from
   * the bits, then a degree-8 odd polynomial in {@code s=(m-1)/(m+1)} on the
   * reduced mantissa. ~1e-12 relative.
   */
  static double log(double x) {
    long bits = Double.doubleToRawLongBits(x);
    int e = (int) ((bits >>> 52) & 0x7ff) - 1023;
    double m = Double.longBitsToDouble((bits & 0x000fffffffffffffL) | 0x3ff0000000000000L);
    if (m > 1.4142135623730951) {   // keep m in [~0.707, ~1.414]
      m *= 0.5;
      e++;
    }
    double s = (m - 1.0) / (m + 1.0);
    double s2 = s * s;
    double poly = s2 * (0.6666666666666735 + s2 * (0.3999999999940942 + s2 * (0.2857142874366239
        + s2 * (0.22222198432149784 + s2 * (0.1818357216161805 + s2 * 0.14798198605116586)))));
    return e * LN2 + 2.0 * s + s * poly;
  }

  /** {@code sin(a)} for {@code a in [0, 2*pi)}. */
  static double sin(double a) {
    return cos(a - HALF_PI);
  }

  /**
   * {@code cos(a)} for {@code a in [0, 2*pi)}: fold to {@code [-pi/4, pi/4]} with
   * a quadrant index, then a degree-6/7 minimax polynomial for sin/cos. ~1e-11.
   */
  static double cos(double a) {
    double y = a * (2.0 / Math.PI);            // quadrants
    int q = (int) (y + (y >= 0 ? 0.5 : -0.5));
    double r = a - q * HALF_PI;                // |r| <= pi/4
    double r2 = r * r;
    double c = 1.0 + r2 * (-0.5 + r2 * (0.041666666666666664 + r2 * (-0.001388888888888889
        + r2 * (2.48015873015873e-5 + r2 * -2.7557319223985893e-7))));
    double s = r * (1.0 + r2 * (-0.16666666666666666 + r2 * (0.008333333333333333
        + r2 * (-1.9841269841269841e-4 + r2 * 2.7557319223985893e-6))));
    return switch (q & 3) {
      case 0 -> c;
      case 1 -> -s;
      case 2 -> -c;
      default -> s;
    };
  }
}
