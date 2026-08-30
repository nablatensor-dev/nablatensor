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
package com.nablatensor.quant;

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.SDouble;
import com.nablatensor.engine.Nabla;
import java.util.function.BiConsumer;

/**
 * Heston stochastic volatility as a composable step block (Seam 5):
 *
 * <pre>{@code
 * dS = r S dt + sqrt(v) S dW1
 * dv = kappa (theta - v) dt + xi sqrt(v) dW2 ,   corr(dW1, dW2) = rho
 * }</pre>
 *
 * <p>Full-truncation Euler: the drift and diffusion of {@code v} use
 * {@code max(v, 0)} while the state may drift slightly negative, which keeps the
 * scheme bias small. Every parameter — including {@code rho} — is a
 * differentiable input, so one adjoint sweep returns the price plus its
 * sensitivity to {@code v0}, {@code kappa}, {@code theta}, {@code xi} and
 * {@code rho} next to the spot / rate Greeks. The {@code max} floor sits on a
 * measure-zero set, so the variance-parameter adjoints match a finite bump only
 * approximately (a few percent); the spot / rate / strike / rho adjoints are
 * exact to Monte-Carlo noise.
 */
public class HestonModel {

  /** Spot and instantaneous variance at a point on the path. */
  public record State(SDouble spot, SDouble variance) {}

  private final SDouble rate;
  private final SDouble kappa;
  private final SDouble theta;
  private final SDouble xi;
  private final SDouble rho;
  private final SDouble rhoBar;   // sqrt(1 - rho^2)
  private final double dt;
  private final double sqrtDt;

  public HestonModel(Nabla.Inputs<HestonMarket> in, double maturity, int steps) {
    this.rate = in.of(HestonMarket::rate);
    this.kappa = in.of(HestonMarket::kappa);
    this.theta = in.of(HestonMarket::theta);
    this.xi = in.of(HestonMarket::xi);
    this.rho = in.of(HestonMarket::rho);
    this.rhoBar = rho.mul(rho).neg().add(1.0).sqrt();
    this.dt = maturity / steps;
    this.sqrtDt = Math.sqrt(dt);
  }

  public State start(Nabla.Inputs<HestonMarket> in) {
    return new State(in.of(HestonMarket::spot), in.of(HestonMarket::v0));
  }

  /**
   * One full-truncation Euler step. {@code z1} drives the spot; {@code zv} is an
   * independent normal that is correlated into the variance factor here.
   */
  public State step(AadRecorder rec, State s, SDouble z1, SDouble zv) {
    SDouble z2 = rho.mul(z1).add(rhoBar.mul(zv));   // corr(z1, z2) = rho

    SDouble vPlus = s.variance().max(0.0);
    SDouble sqrtV = vPlus.sqrt();

    SDouble vNext = s.variance()
        .add(kappa.mul(theta.sub(vPlus)).mul(dt))
        .add(xi.mul(sqrtV).mul(sqrtDt).mul(z2));

    SDouble logDrift = spotDrift(rate.sub(vPlus.mul(0.5)).mul(dt));
    SDouble sNext = s.spot().mul(logDrift.add(spotDiffusion(sqrtV).mul(sqrtDt).mul(z1)).exp());
    return new State(sNext, vNext);
  }

  /** Hook: the per-step spot log-drift. Identity for plain Heston. */
  protected SDouble spotDrift(SDouble base) {
    return base;
  }

  /** Hook: the {@code sqrt(v)} multiplier on the spot Brownian. Identity for plain Heston. */
  protected SDouble spotDiffusion(SDouble sqrtVariance) {
    return sqrtVariance;
  }

  /** A European call/put on the Heston terminal spot, discounted at the flat rate. */
  public static BiConsumer<AadRecorder, Nabla.Inputs<HestonMarket>> european(
      OptionType type, double maturity, int steps) {
    return (rec, in) -> {
      HestonModel m = new HestonModel(in, maturity, steps);
      State s = m.start(in);
      for (int t = 0; t < steps; t++) {
        s = m.step(rec, s, rec.randn(), rec.randn());
      }
      SDouble strike = in.of(HestonMarket::strike);
      SDouble intrinsic = type == OptionType.CALL
          ? s.spot().sub(strike).max(0.0)
          : strike.sub(s.spot()).max(0.0);
      SDouble discount = in.of(HestonMarket::rate).neg().mul(maturity).exp();
      rec.output(intrinsic.mul(discount));
    };
  }
}
