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
import com.nablatensor.ops.SpecialFn;
import java.util.function.BiConsumer;

/**
 * SABR as a composable step block (Seam 5):
 *
 * <pre>{@code
 * dF     = alpha F^beta dW1
 * dalpha = nu alpha dW2 ,   corr(dW1, dW2) = rho
 * }</pre>
 *
 * <p>Euler on the forward (absorbed at zero for the {@code F^beta} term) and the
 * exact log-Euler on {@code alpha}. Used both for a Monte-Carlo price and, via
 * {@link SabrHagan}, as the closed-form target inside {@link Calibrator}.
 *
 * @see SabrMarket
 */
public class SabrModel {

  public record State(SDouble forward, SDouble alpha) {}

  private final SDouble beta;
  private final SDouble nu;
  private final SDouble rho;
  private final SDouble rhoBar;
  private final double dt;
  private final double sqrtDt;

  public SabrModel(Nabla.Inputs<SabrMarket> in, double maturity, int steps) {
    this.beta = in.of(SabrMarket::beta);
    this.nu = in.of(SabrMarket::nu);
    this.rho = in.of(SabrMarket::rho);
    this.rhoBar = rho.mul(rho).neg().add(1.0).sqrt();
    this.dt = maturity / steps;
    this.sqrtDt = Math.sqrt(dt);
  }

  public State start(Nabla.Inputs<SabrMarket> in) {
    return new State(in.of(SabrMarket::forward), in.of(SabrMarket::alpha));
  }

  public State step(AadRecorder rec, State s, SDouble z1, SDouble zv) {
    SDouble z2 = rho.mul(z1).add(rhoBar.mul(zv));
    SDouble fFloor = s.forward().max(1e-8);
    SDouble local = diffusion(s.alpha().mul(SpecialFn.pow(fFloor, beta)));
    SDouble fNext = s.forward().add(local.mul(sqrtDt).mul(z1));
    SDouble aNext = s.alpha().mul(nu.mul(nu).mul(-0.5 * dt).add(nu.mul(sqrtDt).mul(z2)).exp());
    return new State(fNext, aNext);
  }

  /** Hook: the local volatility {@code alpha * F^beta} on the forward. Identity for plain SABR. */
  protected SDouble diffusion(SDouble localVol) {
    return localVol;
  }

  /** European call/put on the terminal forward, discounted at the flat rate. */
  public static BiConsumer<AadRecorder, Nabla.Inputs<SabrMarket>> european(
      OptionType type, double maturity, int steps) {
    return (rec, in) -> {
      SabrModel m = new SabrModel(in, maturity, steps);
      State s = m.start(in);
      for (int t = 0; t < steps; t++) {
        s = m.step(rec, s, rec.randn(), rec.randn());
      }
      SDouble k = in.of(SabrMarket::strike);
      SDouble intrinsic = type == OptionType.CALL
          ? s.forward().sub(k).max(0.0)
          : k.sub(s.forward()).max(0.0);
      SDouble discount = in.of(SabrMarket::rate).neg().mul(maturity).exp();
      rec.output(intrinsic.mul(discount));
    };
  }
}
