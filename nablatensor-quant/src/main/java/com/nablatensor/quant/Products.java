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

/**
 * The MVP catalogue: vanilla European, arithmetic Asian and fixed-strike
 * lookback, each on a {@link GbmPath}. All discounted to today at the flat rate.
 *
 * <p>Each factory returns a {@code Product<EquityMarket>} whose {@code record} is
 * a handful of lines over {@link SDouble} — read one as the template for a payoff
 * of your own. The {@code xxxCall} / {@code xxxPut} pairs are conveniences over
 * the {@link OptionType}-parameterised forms ({@link #european(OptionType)} …),
 * which is the shape to reach for when adding an instrument.
 */
public final class Products {

  private Products() {
  }

  public static Product<EquityMarket> europeanCall() {
    return european(OptionType.CALL);
  }

  public static Product<EquityMarket> europeanPut() {
    return european(OptionType.PUT);
  }

  public static Product<EquityMarket> asianCall() {
    return asian(OptionType.CALL);
  }

  public static Product<EquityMarket> asianPut() {
    return asian(OptionType.PUT);
  }

  public static Product<EquityMarket> lookbackCall() {
    return lookback(OptionType.CALL);
  }

  public static Product<EquityMarket> lookbackPut() {
    return lookback(OptionType.PUT);
  }

  public static Product<EquityMarket> floatingLookbackCall() {
    return floatingLookback(OptionType.CALL);
  }

  public static Product<EquityMarket> floatingLookbackPut() {
    return floatingLookback(OptionType.PUT);
  }

  /** {@code max(sign * (S_T - K), 0)} discounted. Only the terminal value matters. */
  public static Product<EquityMarket> european(OptionType type) {
    return new Named("European " + type, (rec, in, grid) -> {
      Sim sim = new Sim(rec, in, grid);
      SDouble terminal = sim.spot;
      for (int t = 0; t < grid.steps(); t++) {
        terminal = sim.model.step(terminal, rec.randn(), t);
      }
      rec.output(sim.discount(intrinsic(type, terminal, sim.strike)));
    });
  }

  /** {@code max(sign * (mean_t S_t - K), 0)} discounted; arithmetic average over the fixings. */
  public static Product<EquityMarket> asian(OptionType type) {
    return new Named("Asian " + type, (rec, in, grid) -> {
      Sim sim = new Sim(rec, in, grid);
      SDouble path = sim.spot;
      SDouble sum = rec.constant(0.0);
      for (int t = 0; t < grid.steps(); t++) {
        path = sim.model.step(path, rec.randn(), t);
        sum = sum.add(path);
      }
      SDouble average = sum.div((double) grid.steps());
      rec.output(sim.discount(intrinsic(type, average, sim.strike)));
    });
  }

  /**
   * Fixed-strike lookback: {@code max(max_t S_t - K, 0)} for a call,
   * {@code max(K - min_t S_t, 0)} for a put. Discounted. The running extremum
   * includes the initial spot and every simulated fixing.
   */
  public static Product<EquityMarket> lookback(OptionType type) {
    return new Named("Lookback " + type, (rec, in, grid) -> {
      Sim sim = new Sim(rec, in, grid);
      SDouble path = sim.spot;
      SDouble extremum = sim.spot;
      for (int t = 0; t < grid.steps(); t++) {
        path = sim.model.step(path, rec.randn(), t);
        extremum = type == OptionType.CALL ? extremum.max(path) : extremum.min(path);
      }
      SDouble intrinsic = type == OptionType.CALL
          ? extremum.sub(sim.strike).max(0.0)
          : sim.strike.sub(extremum).max(0.0);
      rec.output(sim.discount(intrinsic));
    });
  }

  /**
   * Floating-strike lookback: a call pays {@code S_T - min_t S_t}, a put pays
   * {@code max_t S_t - S_T}. The running extremum includes the initial spot and
   * every simulated fixing. Discounted.
   */
  public static Product<EquityMarket> floatingLookback(OptionType type) {
    return new Named("Floating lookback " + type, (rec, in, grid) -> {
      Sim sim = new Sim(rec, in, grid);
      SDouble path = sim.spot;
      SDouble extremum = sim.spot;
      for (int t = 0; t < grid.steps(); t++) {
        path = sim.model.step(path, rec.randn(), t);
        extremum = type == OptionType.CALL ? extremum.min(path) : extremum.max(path);
      }
      SDouble payoff = type == OptionType.CALL ? path.sub(extremum) : extremum.sub(path);
      rec.output(sim.discount(payoff));   // always non-negative by construction
    });
  }

  private static SDouble intrinsic(OptionType type, SDouble underlying, SDouble strike) {
    SDouble diff = type == OptionType.CALL ? underlying.sub(strike) : strike.sub(underlying);
    return diff.max(0.0);
  }

  /** The recorded market plus a ready {@link GbmPath}; shared setup for every payoff above. */
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
      SDouble vol = in.of(EquityMarket::vol);
      this.model = new GbmPath(rec, rate, vol, grid, maturity);
    }

    SDouble discount(SDouble payoff) {
      return payoff.mul(rate.neg().mul(maturity).exp());
    }
  }

  private record Named(String label, Product<EquityMarket> body) implements Product<EquityMarket> {
    @Override
    public void record(AadRecorder rec, Nabla.Inputs<EquityMarket> in, TimeGrid grid) {
      body.record(rec, in, grid);
    }
  }
}
