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
import com.nablatensor.ops.Smooth;

/**
 * Path-dependent equity payoffs on a GBM path, all monitored with a smoothed
 * indicator (Seam 1 + the {@code nablatensor-ops} smoothing) so the whole payoff
 * stays differentiable and one adjoint sweep gives barrier delta, digital delta,
 * cliquet vega and so on. Shrinking {@code width} recovers the discontinuous
 * contract at the cost of variance.
 */
public final class ExoticProducts {

  /** Up/down and knock-in/out. */
  public enum Barrier { UP_OUT, UP_IN, DOWN_OUT, DOWN_IN }

  private ExoticProducts() {
  }

  /**
   * Single-barrier option with continuous (per-step) smoothed monitoring.
   *
   * @param type    call or put on the terminal spot
   * @param barrier knock level
   * @param kind    up/down × in/out
   * @param width   smoothing width, in spot units (e.g. {@code 0.01 * S0})
   */
  public static Product<EquityMarket> barrier(OptionType type, Barrier kind, double barrier, double width) {
    return new Labelled("Barrier " + kind + " " + type, (rec, in, grid) -> {
      Sim sim = new Sim(rec, in, grid);
      SDouble path = sim.spot;
      SDouble survival = rec.constant(1.0);      // prob(not knocked) so far, smoothed
      boolean up = kind == Barrier.UP_OUT || kind == Barrier.UP_IN;
      for (int t = 0; t < grid.steps(); t++) {
        path = sim.model.step(path, rec.randn(), t);
        SDouble notBreached = up
            ? Smooth.lt(rec, path, barrier, width)   // still below an up-barrier
            : Smooth.gt(rec, path, barrier, width);  // still above a down-barrier
        survival = survival.mul(notBreached);
      }
      SDouble vanilla = type == OptionType.CALL
          ? path.sub(sim.strike).max(0.0)
          : sim.strike.sub(path).max(0.0);
      boolean knockOut = kind == Barrier.UP_OUT || kind == Barrier.DOWN_OUT;
      SDouble alive = knockOut ? survival : rec.constant(1.0).sub(survival);
      rec.output(sim.discount(vanilla.mul(alive)));
    });
  }

  /**
   * Cash-or-nothing digital: pays {@code cash} if the terminal spot finishes in
   * the money, smoothed at the strike.
   */
  public static Product<EquityMarket> digitalCash(OptionType type, double cash, double width) {
    return new Labelled("Digital cash " + type, (rec, in, grid) -> {
      Sim sim = new Sim(rec, in, grid);
      SDouble s = sim.spot;
      for (int t = 0; t < grid.steps(); t++) {
        s = sim.model.step(s, rec.randn(), t);
      }
      SDouble itm = type == OptionType.CALL
          ? Smooth.gt(rec, s, sim.strike, width)
          : Smooth.lt(rec, s, sim.strike, width);
      rec.output(sim.discount(itm.mul(cash)));
    });
  }

  /** Asset-or-nothing digital: pays the terminal spot if it finishes in the money. */
  public static Product<EquityMarket> digitalAsset(OptionType type, double width) {
    return new Labelled("Digital asset " + type, (rec, in, grid) -> {
      Sim sim = new Sim(rec, in, grid);
      SDouble s = sim.spot;
      for (int t = 0; t < grid.steps(); t++) {
        s = sim.model.step(s, rec.randn(), t);
      }
      SDouble itm = type == OptionType.CALL
          ? Smooth.gt(rec, s, sim.strike, width)
          : Smooth.lt(rec, s, sim.strike, width);
      rec.output(sim.discount(itm.mul(s)));
    });
  }

  /**
   * Cliquet / ratchet: each step is a reset period; its return is clamped to
   * {@code [localFloor, localCap]}, the sum is clamped to
   * {@code [globalFloor, globalCap]}, and the notional pays that.
   */
  public static Product<EquityMarket> cliquet(double localFloor, double localCap,
                                double globalFloor, double globalCap, double notional) {
    return new Labelled("Cliquet", (rec, in, grid) -> {
      Sim sim = new Sim(rec, in, grid);
      SDouble prev = sim.spot;
      SDouble sum = rec.constant(0.0);
      for (int t = 0; t < grid.steps(); t++) {
        SDouble next = sim.model.step(prev, rec.randn(), t);
        SDouble ret = next.div(prev).sub(1.0).max(localFloor).min(localCap);
        sum = sum.add(ret);
        prev = next;
      }
      SDouble clamped = sum.max(globalFloor).min(globalCap);
      rec.output(sim.discount(clamped.mul(notional)));
    });
  }

  /**
   * Autocallable note: observed on {@code observations} equally spaced dates. On
   * each date, if the spot is at or above {@code autocallLevel} the note redeems
   * early at par plus an accrued coupon {@code couponPerPeriod} per elapsed
   * observation. If it never triggers, principal redeems with downside
   * {@code min(1, S_T / S0)}. Early redemption is smoothed. Requires
   * {@code steps} to be a multiple of {@code observations}.
   */
  public static Product<EquityMarket> autocallable(double autocallLevel, double couponPerPeriod, int observations,
                                     double width, double notional) {
    return new Labelled("Autocallable", (rec, in, grid) -> {
      int steps = grid.steps();
      if (observations < 1 || steps % observations != 0) {
        throw new IllegalArgumentException("steps (" + steps + ") must be a positive multiple of observations (" + observations + ")");
      }
      int stride = steps / observations;
      Sim sim = new Sim(rec, in, grid);
      SDouble s = sim.spot;
      SDouble alivePrev = rec.constant(1.0);
      SDouble value = rec.constant(0.0);
      SDouble discStep = sim.perStepDiscount(steps);
      SDouble disc = rec.constant(1.0);
      int stepIdx = 0;
      for (int obs = 1; obs <= observations; obs++) {
        for (int k = 0; k < stride; k++) {
          s = sim.model.step(s, rec.randn(), stepIdx++);
          disc = disc.mul(discStep);
        }
        boolean last = obs == observations;
        SDouble triggered = last ? rec.constant(1.0) : Smooth.gt(rec, s, autocallLevel, width);
        SDouble aliveNow = alivePrev.mul(rec.constant(1.0).sub(triggered));
        SDouble redeemedNow = alivePrev.sub(aliveNow);
        SDouble redemption = last
            ? s.div(sim.spot).min(1.0).add(couponPerPeriod * observations)   // principal w/ downside + full coupon
            : rec.constant(1.0 + couponPerPeriod * obs);
        value = value.add(redeemedNow.mul(redemption).mul(notional).mul(disc));
        alivePrev = aliveNow;
      }
      rec.output(value);
    });
  }

  /** Shared GBM setup, mirroring {@link Products}' internal helper. */
  private static final class Sim {
    final SDouble spot;
    final SDouble strike;
    final SDouble rate;
    final SDouble maturity;
    final GbmPath model;

    Sim(AadRecorder rec, Nabla.Inputs<EquityMarket> in, TimeGrid grid) {
      this.spot = in.of(EquityMarket::spot);
      this.strike = in.of(EquityMarket::strike);
      this.rate = in.of(EquityMarket::rate);
      this.maturity = in.of(EquityMarket::maturity);
      this.model = new GbmPath(rec, rate, in.of(EquityMarket::vol), grid, maturity);
    }

    SDouble discount(SDouble payoff) {
      return payoff.mul(rate.neg().mul(maturity).exp());
    }

    SDouble perStepDiscount(int steps) {
      return rate.neg().mul(maturity).div(steps).exp();   // exp(-r T / steps)
    }
  }

  private record Labelled(String label, Product<EquityMarket> body) implements Product<EquityMarket> {
    @Override
    public void record(AadRecorder rec, Nabla.Inputs<EquityMarket> in, TimeGrid grid) {
      body.record(rec, in, grid);
    }
  }
}
