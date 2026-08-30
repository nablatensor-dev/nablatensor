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
 * Hull-White one-factor short rate as a step block (Seam 5):
 *
 * <pre>{@code
 * dr = a (b - r) dt + sigma dW
 * }</pre>
 *
 * <p>A flat mean-reversion level {@code b} stands in for the full
 * {@code theta(t)} term structure — enough to demonstrate rate-model parameter
 * risk ({@code dV/da}, {@code dV/dsigma}, {@code dV/db}) from one adjoint sweep.
 * The path carries a trapezoidal accumulator for {@code integral(r dt)} so a
 * stochastic discount factor is available.
 */
public class HullWhite1F {

  /** Short rate and the running integral of the short rate. */
  public record State(SDouble rate, SDouble integratedRate) {}

  private final SDouble a;
  private final SDouble b;
  private final SDouble sigma;
  private final SDouble r0;
  private final double dt;
  private final double sqrtDt;

  public HullWhite1F(Nabla.Inputs<HullWhiteMarket> in, double maturity, int steps) {
    this.a = in.of(HullWhiteMarket::meanReversion);
    this.b = in.of(HullWhiteMarket::level);
    this.sigma = in.of(HullWhiteMarket::sigma);
    this.r0 = in.of(HullWhiteMarket::r0);
    this.dt = maturity / steps;
    this.sqrtDt = Math.sqrt(dt);
  }

  public State start(AadRecorder rec, Nabla.Inputs<HullWhiteMarket> in) {
    return new State(in.of(HullWhiteMarket::r0), rec.constant(0.0));
  }

  public State step(AadRecorder rec, State s, SDouble z) {
    SDouble rNext = s.rate()
        .add(drift(a.mul(b.sub(s.rate())).mul(dt)))
        .add(diffusion(sigma).mul(sqrtDt).mul(z));
    SDouble accum = s.integratedRate().add(s.rate().add(rNext).mul(0.5 * dt));
    return new State(rNext, accum);
  }

  /** Hook: the per-step deterministic rate move. Identity for plain Hull-White. */
  protected SDouble drift(SDouble meanReversionTerm) {
    return meanReversionTerm;
  }

  /** Hook: the volatility multiplier on the Brownian increment. Identity for plain Hull-White. */
  protected SDouble diffusion(SDouble volatility) {
    return volatility;
  }

  /** {@code exp(-integral(r dt))} along this path. */
  public static SDouble discountFactor(State s) {
    return s.integratedRate().neg().exp();
  }

  /**
   * A caplet on the simulated short rate: {@code notional * tau * max(r_T - K, 0)}
   * paid at {@code T} and discounted along the path. Stylised (the underlying is
   * the short rate itself, not a forward LIBOR) but exercises the full parameter
   * gradient.
   */
  public static BiConsumer<AadRecorder, Nabla.Inputs<HullWhiteMarket>> caplet(
      double maturity, int steps, double tau, double notional) {
    return (rec, in) -> {
      HullWhite1F m = new HullWhite1F(in, maturity, steps);
      State s = m.start(rec, in);
      for (int t = 0; t < steps; t++) {
        s = m.step(rec, s, rec.randn());
      }
      SDouble strike = in.of(HullWhiteMarket::strike);
      SDouble payoff = s.rate().sub(strike).max(0.0).mul(tau * notional);
      rec.output(payoff.mul(discountFactor(s)));
    };
  }

  /** Zero-coupon bond price {@code E[exp(-integral r dt)]}. */
  public static BiConsumer<AadRecorder, Nabla.Inputs<HullWhiteMarket>> zeroCouponBond(
      double maturity, int steps) {
    return (rec, in) -> {
      HullWhite1F m = new HullWhite1F(in, maturity, steps);
      State s = m.start(rec, in);
      for (int t = 0; t < steps; t++) {
        s = m.step(rec, s, rec.randn());
      }
      rec.output(discountFactor(s));
    };
  }

  /**
   * Analytic reconstruction bond {@code P(t, T)} from the simulated short rate
   * {@code r_t}, for an initial curve with a flat instantaneous forward
   * {@code f(0, .) = r0}:
   *
   * <pre>{@code
   * B(t,T) = (1 - e^{-a(T-t)}) / a
   * P(t,T) = exp( -r0 (T-t) + B r0 - sigma^2/(4a) B^2 (1 - e^{-2 a t}) - B r_t )
   * }</pre>
   */
  public SDouble bond(SDouble rt, double t, double horizon) {
    double dtau = horizon - t;
    SDouble bTT = a.mul(-dtau).exp().neg().add(1.0).div(a);              // B(t,T) = (1 - e^{-a dtau}) / a
    SDouble term = sigma.mul(sigma).div(a.mul(4.0))
        .mul(bTT).mul(bTT)
        .mul(a.mul(-2.0 * t).exp().neg().add(1.0));                       // sigma^2/(4a) B^2 (1 - e^{-2 a t})
    return r0.mul(-dtau).add(bTT.mul(r0)).sub(term).sub(bTT.mul(rt)).exp();
  }

  /**
   * Physically-settled European payer swaption expiring at {@code expiry} on an
   * {@code n}-period annual swap struck at {@code K}: at expiry the swap rate and
   * annuity are rebuilt from analytic {@link #bond} prices given the simulated
   * {@code r_T}, and the payoff {@code annuity * max(swapRate - K, 0)} is
   * discounted along the path.
   */
  public static BiConsumer<AadRecorder, Nabla.Inputs<HullWhiteMarket>> europeanSwaption(
      double expiry, int swapPeriods, double accrual, int steps, double strike) {
    return (rec, in) -> {
      HullWhite1F m = new HullWhite1F(in, expiry, steps);
      State s = m.start(rec, in);
      for (int t = 0; t < steps; t++) {
        s = m.step(rec, s, rec.randn());
      }
      SDouble annuity = rec.constant(0.0);
      SDouble lastBond = null;
      for (int i = 1; i <= swapPeriods; i++) {
        SDouble p = m.bond(s.rate(), expiry, expiry + i * accrual);
        annuity = annuity.add(p.mul(accrual));
        lastBond = p;
      }
      SDouble swapRate = rec.constant(1.0).sub(lastBond).div(annuity);
      SDouble payoff = annuity.mul(swapRate.sub(strike).max(0.0));
      rec.output(payoff.mul(discountFactor(s)));
    };
  }
}
