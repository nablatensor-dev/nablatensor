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
 * Characteristic function of the Black-Scholes log-return:
 * {@code phi(u) = exp( i u (r - sigma^2/2) T - sigma^2 u^2 T / 2 )}.
 * The COS price against this must reproduce the closed form to many digits — it
 * is the oracle for the method itself.
 */
public record BsmCf(double rate, double vol) implements CharacteristicFunction {

  @Override
  public Complex phi(double u, double t) {
    double drift = (rate - 0.5 * vol * vol) * t;
    Complex iuDrift = new Complex(0.0, u * drift);
    Complex quad = Complex.real(-0.5 * vol * vol * u * u * t);
    return iuDrift.add(quad).exp();
  }

  @Override
  public double cumulant1(double t) {
    return (rate - 0.5 * vol * vol) * t;
  }

  @Override
  public double cumulant2(double t) {
    return vol * vol * t;
  }
}
