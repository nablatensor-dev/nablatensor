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
import com.nablatensor.engine.ADouble;
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
  public enum BarrierEnum {
    /** BarrierEnum above spot; the option dies if the barrier is touched. */
    UP_OUT,
    /** BarrierEnum above spot; the option only activates if the barrier is touched. */
    UP_IN,
    /** BarrierEnum below spot; the option dies if the barrier is touched. */
    DOWN_OUT,
    /** BarrierEnum below spot; the option only activates if the barrier is touched. */
    DOWN_IN
  }

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
  private static Product<EquityMarket> createBarrier(OptionTypeEnum type, BarrierEnum kind, double barrier, double width) {
    return new Labelled("BarrierEnum " + kind + " " + type, (rec, in, grid) -> {
      Sim sim = new Sim(rec, in, grid);
      ADouble path = sim.spot;
      ADouble survival = rec.constant(1.0);      // prob(not knocked) so far, smoothed
      boolean up = kind == BarrierEnum.UP_OUT || kind == BarrierEnum.UP_IN;
      for (int t = 0; t < grid.steps(); t++) {
        path = sim.model.step(path, rec.randn(), t);
        ADouble notBreached = up
            ? Smooth.lt(rec, path, barrier, width)   // still below an up-barrier
            : Smooth.gt(rec, path, barrier, width);  // still above a down-barrier
        survival = survival.mul(notBreached);
      }
      ADouble vanilla = type == OptionTypeEnum.CALL
          ? path.sub(sim.strike).max(0.0)
          : sim.strike.sub(path).max(0.0);
      boolean knockOut = kind == BarrierEnum.UP_OUT || kind == BarrierEnum.DOWN_OUT;
      ADouble alive = knockOut ? survival : rec.constant(1.0).sub(survival);
      rec.output(sim.discount(vanilla.mul(alive)));
    });
  }

  /**
   * Cash-or-nothing digital: pays {@code cash} if the terminal spot finishes in
   * the money, smoothed at the strike.
   */
  private static Product<EquityMarket> createDigitalCash(OptionTypeEnum type, double cash, double width) {
    return new Labelled("Digital cash " + type, (rec, in, grid) -> {
      Sim sim = new Sim(rec, in, grid);
      ADouble s = sim.spot;
      for (int t = 0; t < grid.steps(); t++) {
        s = sim.model.step(s, rec.randn(), t);
      }
      ADouble itm = type == OptionTypeEnum.CALL
          ? Smooth.gt(rec, s, sim.strike, width)
          : Smooth.lt(rec, s, sim.strike, width);
      rec.output(sim.discount(itm.mul(cash)));
    });
  }

  /** Asset-or-nothing digital: pays the terminal spot if it finishes in the money. */
  private static Product<EquityMarket> createDigitalAsset(OptionTypeEnum type, double width) {
    return new Labelled("Digital asset " + type, (rec, in, grid) -> {
      Sim sim = new Sim(rec, in, grid);
      ADouble s = sim.spot;
      for (int t = 0; t < grid.steps(); t++) {
        s = sim.model.step(s, rec.randn(), t);
      }
      ADouble itm = type == OptionTypeEnum.CALL
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
  private static Product<EquityMarket> createCliquet(double localFloor, double localCap,
                                double globalFloor, double globalCap, double notional) {
    return new Labelled("Cliquet", (rec, in, grid) -> {
      Sim sim = new Sim(rec, in, grid);
      ADouble prev = sim.spot;
      ADouble sum = rec.constant(0.0);
      for (int t = 0; t < grid.steps(); t++) {
        ADouble next = sim.model.step(prev, rec.randn(), t);
        ADouble ret = next.div(prev).sub(1.0).max(localFloor).min(localCap);
        sum = sum.add(ret);
        prev = next;
      }
      ADouble clamped = sum.max(globalFloor).min(globalCap);
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
  private static Product<EquityMarket> createAutocallable(double autocallLevel, double couponPerPeriod, int observations,
                                     double width, double notional) {
    return new Labelled("Autocallable", (rec, in, grid) -> {
      int steps = grid.steps();
      if (observations < 1 || steps % observations != 0) {
        throw new IllegalArgumentException("steps (" + steps + ") must be a positive multiple of observations (" + observations + ")");
      }
      int stride = steps / observations;
      Sim sim = new Sim(rec, in, grid);
      ADouble s = sim.spot;
      ADouble alivePrev = rec.constant(1.0);
      ADouble value = rec.constant(0.0);
      ADouble discStep = sim.perStepDiscount(steps);
      ADouble disc = rec.constant(1.0);
      int stepIdx = 0;
      for (int obs = 1; obs <= observations; obs++) {
        for (int k = 0; k < stride; k++) {
          s = sim.model.step(s, rec.randn(), stepIdx++);
          disc = disc.mul(discStep);
        }
        boolean last = obs == observations;
        ADouble triggered = last ? rec.constant(1.0) : Smooth.gt(rec, s, autocallLevel, width);
        ADouble aliveNow = alivePrev.mul(rec.constant(1.0).sub(triggered));
        ADouble redeemedNow = alivePrev.sub(aliveNow);
        ADouble redemption = last
            ? s.div(sim.spot).min(1.0).add(couponPerPeriod * observations)   // principal w/ downside + full coupon
            : rec.constant(1.0 + couponPerPeriod * obs);
        value = value.add(redeemedNow.mul(redemption).mul(notional).mul(disc));
        alivePrev = aliveNow;
      }
      rec.output(value);
    });
  }


  /** Named construction for a single-barrier option. */
  public static final class BarrierOption {
    private OptionTypeEnum type;
    private BarrierEnum kind;
    private Double barrier;
    private Double width;

    private BarrierOption() {}
    public static BarrierOption of() { return new BarrierOption(); }
    public BarrierOption type(OptionTypeEnum value) { type = value; return this; }
    public BarrierOption kind(BarrierEnum value) { kind = value; return this; }
    public BarrierOption barrier(double value) { barrier = value; return this; }
    public BarrierOption width(double value) { width = value; return this; }

    public Product<EquityMarket> build() {
      return ExoticProducts.createBarrier(required(type, "type"), required(kind, "kind"),
          required(barrier, "barrier"), required(width, "width"));
    }
  }

  /** Named construction for a cash-or-nothing digital option. */
  public static final class DigitalCash {
    private OptionTypeEnum type;
    private Double cash;
    private Double width;

    private DigitalCash() {}
    public static DigitalCash of() { return new DigitalCash(); }
    public DigitalCash type(OptionTypeEnum value) { type = value; return this; }
    public DigitalCash cash(double value) { cash = value; return this; }
    public DigitalCash width(double value) { width = value; return this; }

    public Product<EquityMarket> build() {
      return createDigitalCash(required(type, "type"), required(cash, "cash"), required(width, "width"));
    }
  }

  /** Named construction for an asset-or-nothing digital option. */
  public static final class DigitalAsset {
    private OptionTypeEnum type;
    private Double width;

    private DigitalAsset() {}
    public static DigitalAsset of() { return new DigitalAsset(); }
    public DigitalAsset type(OptionTypeEnum value) { type = value; return this; }
    public DigitalAsset width(double value) { width = value; return this; }

    public Product<EquityMarket> build() {
      return createDigitalAsset(required(type, "type"), required(width, "width"));
    }
  }

  /** Named construction for a cliquet/ratchet payoff. */
  public static final class Cliquet {
    private Double localFloor;
    private Double localCap;
    private Double globalFloor;
    private Double globalCap;
    private Double notional;

    private Cliquet() {}
    public static Cliquet of() { return new Cliquet(); }
    public Cliquet localFloor(double value) { localFloor = value; return this; }
    public Cliquet localCap(double value) { localCap = value; return this; }
    public Cliquet globalFloor(double value) { globalFloor = value; return this; }
    public Cliquet globalCap(double value) { globalCap = value; return this; }
    public Cliquet notional(double value) { notional = value; return this; }

    public Product<EquityMarket> build() {
      return createCliquet(required(localFloor, "localFloor"), required(localCap, "localCap"),
          required(globalFloor, "globalFloor"), required(globalCap, "globalCap"),
          required(notional, "notional"));
    }
  }

  /** Named construction for an autocallable note. */
  public static final class Autocallable {
    private Double autocallLevel;
    private Double couponPerPeriod;
    private Integer observations;
    private Double width;
    private Double notional;

    private Autocallable() {}
    public static Autocallable of() { return new Autocallable(); }
    public Autocallable autocallLevel(double value) { autocallLevel = value; return this; }
    public Autocallable couponPerPeriod(double value) { couponPerPeriod = value; return this; }
    public Autocallable observations(int value) { observations = value; return this; }
    public Autocallable width(double value) { width = value; return this; }
    public Autocallable notional(double value) { notional = value; return this; }

    public Product<EquityMarket> build() {
      return createAutocallable(required(autocallLevel, "autocallLevel"),
          required(couponPerPeriod, "couponPerPeriod"), required(observations, "observations"),
          required(width, "width"), required(notional, "notional"));
    }
  }

  private static <T> T required(T value, String name) {
    if (value == null) {
      throw new IllegalStateException("Required field " + name + " is not set");
    }
    return value;
  }

  /** Shared GBM setup, mirroring {@link Products}' internal helper. */

  private static final class Sim {
    final ADouble spot;
    final ADouble strike;
    final ADouble rate;
    final ADouble maturity;
    final GbmPath model;

    Sim(AadRecorder rec, Nabla.Inputs<EquityMarket> in, TimeGrid grid) {
      this.spot = in.of(EquityMarket::spot);
      this.strike = in.of(EquityMarket::strike);
      this.rate = in.of(EquityMarket::rate);
      this.maturity = in.of(EquityMarket::maturity);
      this.model = GbmPath.of(rec, rate, in.of(EquityMarket::vol), grid, maturity);
    }

    ADouble discount(ADouble payoff) {
      return payoff.mul(rate.neg().mul(maturity).exp());
    }

    ADouble perStepDiscount(int steps) {
      return rate.neg().mul(maturity).div(steps).exp();   // exp(-r T / steps)
    }
  }

  private record Labelled(String label, Product<EquityMarket> body) implements Product<EquityMarket> {
    @Override
    public void record(AadRecorder rec, Nabla.Inputs<EquityMarket> in, TimeGrid grid) {
      body.record(rec, in, grid);
    }

    @Override
    public String toString() {
      return label;
    }
  }
}
