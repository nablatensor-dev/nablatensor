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
package com.nablatensor.cva;

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.SDouble;

/**
 * A one-factor Hull-White short rate as an on-tape step block, plus an analytic
 * zero-coupon bond reconstruction {@code P(t, T)} from the simulated {@code r_t}.
 *
 * <pre>{@code
 * dr = a (b - r) dt + sigma dW
 * }</pre>
 *
 * <p>A flat instantaneous forward {@code f(0, .) = r0} stands in for the full
 * {@code theta(t)} term structure — the same simplification as
 * {@code com.nablatensor.quant.HullWhite1F#bond}, which this mirrors. Every
 * coefficient is an {@link SDouble} input, so one adjoint sweep of a valuation
 * built on this block yields {@code dV/dr0}, {@code dV/da}, {@code dV/db} and
 * {@code dV/dsigma} together.
 */
public final class HwShortRate {

  /** The simulated short rate and the running trapezoidal integral of it. */
  public record State(SDouble rate, SDouble integratedRate) {}

  private final AadRecorder recorder;
  private final SDouble r0;
  private final SDouble level;
  private final SDouble meanReversion;
  private final SDouble sigma;
  private final double dt;
  private final double sqrtDt;

  public HwShortRate(AadRecorder recorder, SDouble r0, SDouble level,
                     SDouble meanReversion, SDouble sigma, double dt) {
    if (!(dt > 0.0)) {
      throw new IllegalArgumentException("dt must be > 0, got " + dt);
    }
    this.recorder = recorder;
    this.r0 = r0;
    this.level = level;
    this.meanReversion = meanReversion;
    this.sigma = sigma;
    this.dt = dt;
    this.sqrtDt = Math.sqrt(dt);
  }

  /** The path at {@code t = 0}: {@code r = r0}, integral {@code 0}. */
  public State start() {
    return new State(r0, recorder.constant(0.0));
  }

  /** One Euler step forward given a standard-normal draw {@code z}. */
  public State step(State s, SDouble z) {
    SDouble rNext = s.rate()
        .add(meanReversion.mul(level.sub(s.rate())).mul(dt))
        .add(sigma.mul(sqrtDt).mul(z));
    SDouble accum = s.integratedRate().add(s.rate().add(rNext).mul(0.5 * dt));
    return new State(rNext, accum);
  }

  /** {@code exp(-integral_0^t r du)} along this path — the stochastic discount factor. */
  public SDouble discountFactor(State s) {
    return s.integratedRate().neg().exp();
  }

  /**
   * Analytic {@code P(t, T)} from the simulated short rate {@code r_t}, for an
   * initial curve with a flat instantaneous forward {@code f(0, .) = r0}:
   *
   * <pre>{@code
   * B(t,T) = (1 - e^{-a (T-t)}) / a
   * P(t,T) = exp( -r0 (T-t) + B r0 - sigma^2/(4a) B^2 (1 - e^{-2 a t}) - B r_t )
   * }</pre>
   */
  public SDouble bond(SDouble rt, double t, double horizon) {
    double dtau = horizon - t;
    if (dtau <= 0.0) {
      return recorder.constant(1.0);
    }
    SDouble b = meanReversion.mul(-dtau).exp().neg().add(1.0).div(meanReversion);
    SDouble term = sigma.mul(sigma).div(meanReversion.mul(4.0))
        .mul(b).mul(b)
        .mul(meanReversion.mul(-2.0 * t).exp().neg().add(1.0));
    return r0.mul(-dtau).add(b.mul(r0)).sub(term).sub(b.mul(rt)).exp();
  }

  AadRecorder recorder() {
    return recorder;
  }
}
