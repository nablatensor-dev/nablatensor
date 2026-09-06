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
import com.nablatensor.ops.Smooth;
import java.util.function.BiConsumer;

/**
 * Merton jump-diffusion as a composable step block (Seam 5):
 *
 * <pre>{@code
 * dS/S = (r - lambda kappa) dt + sigma dW + (Y - 1) dN
 * ln Y ~ N(muJ, deltaJ^2)
 * kappa = E[Y - 1] = exp(muJ + deltaJ^2 / 2) - 1
 * }</pre>
 *
 * <p>Each step draws a diffusion normal and, from a uniform, a smoothed
 * indicator {@code 1{U < lambda dt}} for at most one jump (exact as the
 * monitoring gets fine, since {@code P(>= 2 jumps per step) = O((lambda dt)^2)}).
 * The jump count is not differentiated; {@code muJ}, {@code deltaJ} and the
 * compensator are, so one adjoint sweep returns the jump-parameter risk next to
 * the spot / vol / rate Greeks.
 *
 * <p>Validated against the exact
 * {@link com.nablatensor.quant.analytic.MertonJumpDiffusion} Poisson-series price
 * (agrees to Monte-Carlo error). The smoothed jump indicator leaves an
 * {@code O(0.1%)} martingale bias in {@code E[S_T]} — negligible for option
 * prices, visible only in a put-call-parity check at high path counts — that
 * shrinks with the indicator width.
 */
public class MertonJumpModel {

  private final SDouble rate;
  private final SDouble vol;
  private final SDouble intensity;
  private final SDouble jumpMean;
  private final SDouble jumpVol;
  private final SDouble kappa;
  private final double dt;
  private final double sqrtDt;
  private final double indicatorWidth;

  public MertonJumpModel(Nabla.Inputs<MertonJumpMarket> in, double maturity, int steps,
                         double indicatorWidth) {
    this.rate = in.of(MertonJumpMarket::rate);
    this.vol = in.of(MertonJumpMarket::vol);
    this.intensity = in.of(MertonJumpMarket::jumpIntensity);
    this.jumpMean = in.of(MertonJumpMarket::jumpMean);
    this.jumpVol = in.of(MertonJumpMarket::jumpVol);
    this.kappa = jumpMean.add(jumpVol.mul(jumpVol).mul(0.5)).exp().sub(1.0);
    this.dt = maturity / steps;
    this.sqrtDt = Math.sqrt(dt);
    this.indicatorWidth = indicatorWidth;
  }

  public SDouble start(Nabla.Inputs<MertonJumpMarket> in) {
    return in.of(MertonJumpMarket::spot);
  }

  /**
   * One step. {@code z} drives the diffusion, {@code u} selects whether a jump
   * occurs this step, {@code zJump} is its (log) size.
   */
  public SDouble step(AadRecorder rec, SDouble s, SDouble z, SDouble u, SDouble zJump) {
    // Bernoulli jump factor has expectation (1 + lambda kappa dt), so the exact
    // martingale compensator for this scheme is -ln(1 + lambda kappa dt), not
    // -lambda kappa dt (they agree to O(dt^2)).
    SDouble compensator = intensity.mul(kappa).mul(dt).add(1.0).log();
    SDouble drift = rate.sub(vol.mul(vol).mul(0.5)).mul(dt).sub(compensator);
    SDouble diffused = s.mul(drift.add(vol.mul(sqrtDt).mul(z)).exp());

    SDouble p = intensity.mul(dt);
    SDouble jumpOccurs = Smooth.lt(rec, u, p, indicatorWidth);      // ~ Bernoulli(lambda dt)
    SDouble jumpFactor = jumpMean.add(jumpVol.mul(zJump)).exp().sub(1.0);
    return diffused.mul(rec.constant(1.0).add(jumpOccurs.mul(jumpFactor)));
  }

  /** European call/put on the terminal spot, discounted at the flat rate. */
  public static BiConsumer<AadRecorder, Nabla.Inputs<MertonJumpMarket>> european(
      OptionType type, double maturity, int steps) {
    return european(type, maturity, steps, 2.0e-4);
  }

  public static BiConsumer<AadRecorder, Nabla.Inputs<MertonJumpMarket>> european(
      OptionType type, double maturity, int steps, double indicatorWidth) {
    return (rec, in) -> {
      MertonJumpModel m = new MertonJumpModel(in, maturity, steps, indicatorWidth);
      SDouble s = m.start(in);
      for (int t = 0; t < steps; t++) {
        s = m.step(rec, s, rec.randn(), rec.randu(), rec.randn());
      }
      SDouble strike = in.of(MertonJumpMarket::strike);
      SDouble intrinsic = type == OptionType.CALL ? s.sub(strike).max(0.0) : strike.sub(s).max(0.0);
      SDouble discount = in.of(MertonJumpMarket::rate).neg().mul(maturity).exp();
      rec.output(intrinsic.mul(discount));
    };
  }
}
