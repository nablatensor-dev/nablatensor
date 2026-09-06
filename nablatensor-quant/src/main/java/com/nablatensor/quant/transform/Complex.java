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
package com.nablatensor.quant.transform;

/**
 * A minimal complex number for the characteristic-function evaluations in this
 * package. Principal branches for {@link #log} and {@link #sqrt}; the Heston
 * characteristic function is written in the branch-stable "little Heston trap"
 * form so these suffice.
 */
public record Complex(double re, double im) {

  public static final Complex ZERO = new Complex(0, 0);
  public static final Complex ONE = new Complex(1, 0);
  public static final Complex I = new Complex(0, 1);

  public static Complex real(double x) {
    return new Complex(x, 0);
  }

  public Complex add(Complex o) {
    return new Complex(re + o.re, im + o.im);
  }

  public Complex add(double x) {
    return new Complex(re + x, im);
  }

  public Complex sub(Complex o) {
    return new Complex(re - o.re, im - o.im);
  }

  public Complex mul(Complex o) {
    return new Complex(re * o.re - im * o.im, re * o.im + im * o.re);
  }

  public Complex mul(double x) {
    return new Complex(re * x, im * x);
  }

  public Complex div(Complex o) {
    double d = o.re * o.re + o.im * o.im;
    return new Complex((re * o.re + im * o.im) / d, (im * o.re - re * o.im) / d);
  }

  public Complex neg() {
    return new Complex(-re, -im);
  }

  public double abs() {
    return Math.hypot(re, im);
  }

  /** Principal-branch natural logarithm. */
  public Complex log() {
    return new Complex(Math.log(abs()), Math.atan2(im, re));
  }

  /** Principal-branch square root. */
  public Complex sqrt() {
    double r = abs();
    double sr = Math.sqrt((r + re) / 2.0);
    double si = Math.sqrt((r - re) / 2.0);
    return new Complex(sr, im < 0 ? -si : si);
  }

  /** {@code exp(z)}. */
  public Complex exp() {
    double e = Math.exp(re);
    return new Complex(e * Math.cos(im), e * Math.sin(im));
  }

  /** {@code z^p} for a real exponent, via {@code exp(p log z)}. */
  public Complex pow(double p) {
    return log().mul(p).exp();
  }
}
