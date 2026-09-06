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
import com.nablatensor.engine.Nabla;
import com.nablatensor.engine.SDouble;
import java.util.function.BiConsumer;

/**
 * Schwartz one-factor mean-reverting commodity spot as a composable step block
 * (Seam 5) — the same Ornstein-Uhlenbeck machinery as {@link HullWhite1F},
 * applied to the log price:
 *
 * <pre>{@code
 * d ln S = kappa (level - ln S) dt + sigma dW
 * }</pre>
 *
 * <p>The log price is Gaussian, so at horizon {@code T}:
 * <ul>
 *   <li>{@code E[ln S_T] = e^{-kappa T} ln S_0 + (1 - e^{-kappa T}) level};</li>
 *   <li>{@code Var[ln S_T] = sigma^2 (1 - e^{-2 kappa T}) / (2 kappa)}, which
 *       tends to the stationary {@code sigma^2 / (2 kappa)};</li>
 *   <li>the futures price is {@link #futuresPrice} below.</li>
 * </ul>
 */
public class SchwartzOneFactor {

  private final SDouble kappa;
  private final SDouble level;
  private final SDouble sigma;
  private final double dt;
  private final double sqrtDt;

  public SchwartzOneFactor(Nabla.Inputs<SchwartzMarket> in, double maturity, int steps) {
    this.kappa = in.of(SchwartzMarket::kappa);
    this.level = in.of(SchwartzMarket::level);
    this.sigma = in.of(SchwartzMarket::sigma);
    this.dt = maturity / steps;
    this.sqrtDt = Math.sqrt(dt);
  }

  public SDouble startLog(Nabla.Inputs<SchwartzMarket> in) {
    return in.of(SchwartzMarket::spot).log();
  }

  /** One Euler step on the log price. */
  public SDouble step(SDouble logSpot, SDouble z) {
    return logSpot
        .add(kappa.mul(level.sub(logSpot)).mul(dt))
        .add(sigma.mul(sqrtDt).mul(z));
  }

  /**
   * The closed-form futures price today for delivery at {@code T}:
   * {@code exp( e^{-kappa T} ln S_0 + (1 - e^{-kappa T}) level
   *            + sigma^2 (1 - e^{-2 kappa T}) / (4 kappa) )}.
   */
  public static double futuresPrice(SchwartzMarket m, double maturity) {
    double ek = Math.exp(-m.kappa() * maturity);
    double e2k = Math.exp(-2.0 * m.kappa() * maturity);
    double meanLog = ek * Math.log(m.spot()) + (1.0 - ek) * m.level();
    double varLog = m.sigma() * m.sigma() * (1.0 - e2k) / (2.0 * m.kappa());
    return Math.exp(meanLog + 0.5 * varLog);
  }

  /**
   * European call/put on the terminal spot, discounted at the flat rate. Not a
   * futures option — the underlying is {@code S_T} itself.
   */
  public static BiConsumer<AadRecorder, Nabla.Inputs<SchwartzMarket>> european(
      OptionType type, double strike, double maturity, int steps) {
    return (rec, in) -> {
      SchwartzOneFactor m = new SchwartzOneFactor(in, maturity, steps);
      SDouble logS = m.startLog(in);
      for (int t = 0; t < steps; t++) {
        logS = m.step(logS, rec.randn());
      }
      SDouble s = logS.exp();
      SDouble intrinsic = type == OptionType.CALL ? s.sub(strike).max(0.0) : rec.constant(strike).sub(s).max(0.0);
      SDouble discount = in.of(SchwartzMarket::rate).neg().mul(maturity).exp();
      rec.output(intrinsic.mul(discount));
    };
  }
}
