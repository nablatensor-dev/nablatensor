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
import com.nablatensor.engine.ADouble;
import com.nablatensor.ops.Smooth;
import java.util.function.BiConsumer;

/**
 * Kou double-exponential jump-diffusion as a composable step block (Seam 5).
 *
 * <p>Structure follows {@link MertonJumpModel}: one smoothed at-most-one-jump
 * indicator per step, but the log jump size is drawn from the asymmetric
 * two-sided exponential by inverting each side's CDF from a uniform. The
 * risk-neutral compensator is
 *
 * <pre>{@code
 * kappa = probUp * etaUp / (etaUp - 1) + (1 - probUp) * etaDown / (etaDown + 1) - 1
 * }</pre>
 *
 * <p>{@code probUp}, {@code etaUp}, {@code etaDown} are differentiable inputs.
 */
public class KouJumpModel {

  private final ADouble rate;
  private final ADouble vol;
  private final ADouble intensity;
  private final ADouble probUp;
  private final ADouble etaUp;
  private final ADouble etaDown;
  private final ADouble kappa;
  private final double dt;
  private final double sqrtDt;
  private final double indicatorWidth;

  public static KouJumpModel of(Nabla.Inputs<KouMarket> in, double maturity, int steps, double indicatorWidth) {
    return new KouJumpModel(in, maturity, steps, indicatorWidth);
  }

  protected KouJumpModel(Nabla.Inputs<KouMarket> in, double maturity, int steps, double indicatorWidth) {
    this.rate = in.of(KouMarket::rate);
    this.vol = in.of(KouMarket::vol);
    this.intensity = in.of(KouMarket::jumpIntensity);
    this.probUp = in.of(KouMarket::probUp);
    this.etaUp = in.of(KouMarket::etaUp);
    this.etaDown = in.of(KouMarket::etaDown);
    ADouble up = probUp.mul(etaUp.div(etaUp.sub(1.0)));
    ADouble down = rec1(probUp).mul(etaDown.div(etaDown.add(1.0)));
    this.kappa = up.add(down).sub(1.0);
    this.dt = maturity / steps;
    this.sqrtDt = Math.sqrt(dt);
    this.indicatorWidth = indicatorWidth;
  }

  private static ADouble rec1(ADouble p) {
    return p.neg().add(1.0);
  }

  public ADouble start(Nabla.Inputs<KouMarket> in) {
    return in.of(KouMarket::spot);
  }

  /**
   * One step. {@code z} drives the diffusion, {@code uJump} selects whether a
   * jump occurs, {@code uSide} selects up vs down, {@code uMag} is inverted to
   * the exponential magnitude.
   */
  public ADouble step(AadRecorder rec, ADouble s, ADouble z,
                      ADouble uJump, ADouble uSide, ADouble uMag) {
    // Exact martingale compensator for the at-most-one-jump-per-step scheme
    // (jump factor expectation 1 + lambda kappa dt): -ln(1 + lambda kappa dt).
    ADouble compensator = intensity.mul(kappa).mul(dt).add(1.0).log();
    ADouble drift = rate.sub(vol.mul(vol).mul(0.5)).mul(dt).sub(compensator);
    ADouble diffused = s.mul(drift.add(vol.mul(sqrtDt).mul(z)).exp());

    ADouble p = intensity.mul(dt);
    ADouble jumpOccurs = Smooth.lt(rec, uJump, p, indicatorWidth);
    ADouble isUp = Smooth.lt(rec, uSide, probUp, 1.0e-3);
    // Exp(eta) magnitude from a uniform: -ln(1 - u) / eta.
    ADouble expMag = rec.constant(1.0).sub(uMag).log().neg();
    ADouble jUp = expMag.div(etaUp);
    ADouble jDown = expMag.div(etaDown).neg();
    ADouble logJump = isUp.mul(jUp).add(rec1(isUp).mul(jDown));

    ADouble jumpFactor = logJump.exp().sub(1.0);
    return diffused.mul(rec.constant(1.0).add(jumpOccurs.mul(jumpFactor)));
  }

  public static BiConsumer<AadRecorder, Nabla.Inputs<KouMarket>> european(
      OptionTypeEnum type, double maturity, int steps) {
    return european(type, maturity, steps, 2.0e-4);
  }

  public static BiConsumer<AadRecorder, Nabla.Inputs<KouMarket>> european(
      OptionTypeEnum type, double maturity, int steps, double indicatorWidth) {
    return (rec, in) -> {
      KouJumpModel m = KouJumpModel.of(in, maturity, steps, indicatorWidth);
      ADouble s = m.start(in);
      for (int t = 0; t < steps; t++) {
        s = m.step(rec, s, rec.randn(), rec.randu(), rec.randu(), rec.randu());
      }
      ADouble strike = in.of(KouMarket::strike);
      ADouble intrinsic = type == OptionTypeEnum.CALL ? s.sub(strike).max(0.0) : strike.sub(s).max(0.0);
      ADouble discount = in.of(KouMarket::rate).neg().mul(maturity).exp();
      rec.output(intrinsic.mul(discount));
    };
  }
}
