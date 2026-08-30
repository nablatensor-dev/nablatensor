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
 * A four-factor LIBOR Market Model (BGM) strip as a step block (Seam 5).
 *
 * <p>Four forward rates {@code L1..L4} over equal accrual periods of length
 * {@code tenor}, evolved log-normally under the terminal measure with a flat
 * instantaneous volatility and a flat Brownian correlation:
 *
 * <pre>{@code
 * dL_i / L_i = mu_i dt + vol dW_i ,   corr(dW_i, dW_j) = corr
 * mu_i       = - vol * sum_{j>i} corr * tenor * vol * L_j / (1 + tenor L_j)
 * }</pre>
 *
 * <p>Terminal measure (numeraire = bond maturing after the last period), so the
 * drift of the last forward is zero and the others carry the usual
 * frozen-per-step summation. {@code vol} and {@code corr} are differentiable
 * inputs. The demo payoff is a physically-settled payer swaption on the strip.
 */
public final class LmmModel {

  public static final int RATES = 4;

  private final double tenor;
  private final double dt;
  private final double sqrtDt;
  private final SDouble vol;
  private final SDouble corr;

  public LmmModel(Nabla.Inputs<LmmMarket> in, double expiry, int steps) {
    this.tenor = LmmMarket.TENOR;
    this.dt = expiry / steps;
    this.sqrtDt = Math.sqrt(dt);
    this.vol = in.of(LmmMarket::vol);
    this.corr = in.of(LmmMarket::corr);
  }

  public SDouble[] start(Nabla.Inputs<LmmMarket> in) {
    return new SDouble[] {
        in.of(LmmMarket::l1), in.of(LmmMarket::l2), in.of(LmmMarket::l3), in.of(LmmMarket::l4)};
  }

  /** One log-Euler step with a frozen terminal-measure drift; {@code z[i]} correlated already. */
  public SDouble[] step(AadRecorder rec, SDouble[] l, SDouble[] z) {
    SDouble[] next = new SDouble[RATES];
    for (int i = 0; i < RATES; i++) {
      SDouble drift = rec.constant(0.0);
      for (int j = i + 1; j < RATES; j++) {
        SDouble term = l[j].mul(tenor).mul(vol).mul(corr).div(l[j].mul(tenor).add(1.0));
        drift = drift.sub(vol.mul(term));
      }
      SDouble logIncr = drift.sub(vol.mul(vol).mul(0.5)).mul(dt).add(vol.mul(sqrtDt).mul(z[i]));
      next[i] = l[i].mul(logIncr.exp());
    }
    return next;
  }

  /**
   * Per-step innovations. This minimal build uses independent innovations and
   * carries the flat {@code corr} only through the drift summation; a full
   * SDouble Cholesky of the innovations (so {@code dV/dcorr} also picks up the
   * diffusion channel) is a later refinement.
   */
  public SDouble[] draw(AadRecorder rec) {
    SDouble[] z = new SDouble[RATES];
    for (int i = 0; i < RATES; i++) {
      z[i] = rec.randn();
    }
    return z;
  }

  /**
   * Payer swaption struck at {@code K}, expiring after {@code steps} sub-steps,
   * on the whole four-period strip; settled with the model's own annuity.
   */
  public static BiConsumer<AadRecorder, Nabla.Inputs<LmmMarket>> payerSwaption(
      double expiry, int steps, double strike) {
    return (rec, in) -> {
      LmmModel m = new LmmModel(in, expiry, steps);
      SDouble[] l = m.start(in);
      for (int t = 0; t < steps; t++) {
        l = m.step(rec, l, m.draw(rec));
      }
      // annuity A = sum tenor * P(0, T_{i+1}); P built forward from the simulated forwards
      SDouble discount = rec.constant(1.0);
      SDouble annuity = rec.constant(0.0);
      SDouble floatingLeg = rec.constant(0.0);
      for (int i = 0; i < RATES; i++) {
        discount = discount.div(l[i].mul(LmmMarket.TENOR).add(1.0));
        annuity = annuity.add(discount.mul(LmmMarket.TENOR));
        floatingLeg = floatingLeg.add(discount.mul(l[i]).mul(LmmMarket.TENOR));
      }
      SDouble swapRate = floatingLeg.div(annuity);
      rec.output(swapRate.sub(strike).max(0.0).mul(annuity));
    };
  }

  /** Receiver swaption: {@code annuity * max(K - swapRate, 0)}. */
  public static BiConsumer<AadRecorder, Nabla.Inputs<LmmMarket>> receiverSwaption(
      double expiry, int steps, double strike) {
    return (rec, in) -> {
      LmmModel m = new LmmModel(in, expiry, steps);
      SDouble[] l = m.start(in);
      for (int t = 0; t < steps; t++) {
        l = m.step(rec, l, m.draw(rec));
      }
      SDouble discount = rec.constant(1.0);
      SDouble annuity = rec.constant(0.0);
      SDouble floatingLeg = rec.constant(0.0);
      for (int i = 0; i < RATES; i++) {
        discount = discount.div(l[i].mul(LmmMarket.TENOR).add(1.0));
        annuity = annuity.add(discount.mul(LmmMarket.TENOR));
        floatingLeg = floatingLeg.add(discount.mul(l[i]).mul(LmmMarket.TENOR));
      }
      SDouble swapRate = floatingLeg.div(annuity);
      rec.output(annuity.mul(swapRate.neg().add(strike).max(0.0)));
    };
  }

  /**
   * Cap ({@code CALL}) or floor ({@code PUT}) on the four-period strip: the strip
   * is evolved {@code stepsPerPeriod} sub-steps per accrual period; forward
   * {@code L_i} is read at the start of period {@code i} as its reset, and the
   * caplet {@code tenor * max(sign (L_i - K), 0)} pays at the period end,
   * discounted with the model's own reconstructed factors.
   */
  public static BiConsumer<AadRecorder, Nabla.Inputs<LmmMarket>> capFloor(
      OptionType type, int stepsPerPeriod, double strike, double notional) {
    return (rec, in) -> {
      double sign = type.sign();
      LmmModel stepper = new LmmModel(in, LmmMarket.TENOR, Math.max(1, stepsPerPeriod));
      SDouble[] l = stepper.start(in);
      SDouble discount = rec.constant(1.0);
      SDouble value = rec.constant(0.0);
      for (int i = 0; i < RATES; i++) {
        SDouble caplet = l[i].sub(strike).mul(sign).max(0.0).mul(LmmMarket.TENOR * notional);
        discount = discount.div(l[i].mul(LmmMarket.TENOR).add(1.0));   // now to end of period i
        value = value.add(caplet.mul(discount));
        if (i < RATES - 1) {
          for (int s = 0; s < Math.max(1, stepsPerPeriod); s++) {
            l = stepper.step(rec, l, stepper.draw(rec));
          }
        }
      }
      rec.output(value);
    };
  }
}
