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
package com.nablatensor.cva;

import com.nablatensor.codegen.Of;

import com.nablatensor.engine.ADouble;
import java.util.Locale;

/**
 * A single-settlement FX forward: exchange {@code foreignNotional} of foreign
 * currency for {@code strike} reporting-currency per unit at {@code settlementYears}.
 * At a grid date {@code t} the forward FX rate is rebuilt from the simulated FX
 * spot and the two discount curves,
 *
 * <pre>{@code
 * F(t, T) = fxSpot(t) * P_foreign(t, T) / P_domestic(t, T)
 * value(t) = side * foreignNotional * (F(t, T) - strike) * P_domestic(t, T)
 * }</pre>
 *
 * so the trade contributes FX delta and FX-vega to the netting-set CVA gradient
 * alongside the swaps' rate risk.
 *
 * @param id              trade id
 * @param side            buy or sell the foreign currency
 * @param foreignNotional foreign-currency notional
 * @param strike          agreed rate, reporting currency per unit foreign
 * @param settlementYears settlement date in years
 */
@Of
public final class FxForward implements CvaTrade {

  private final String id;
  private final SideEnum side;
  private final double foreignNotional;
  private final double strike;
  private final double settlementYears;
  static FxForward create(String id, SideEnum side, double foreignNotional, double strike, double settlementYears) {
    return new FxForward(id, side, foreignNotional, strike, settlementYears);
  }

  public static FxForwardBuilder of() {
    return new FxForwardBuilder();
  }

  public String id() {
    return id;
  }

  public SideEnum side() {
    return side;
  }

  public double foreignNotional() {
    return foreignNotional;
  }

  public double strike() {
    return strike;
  }

  public double settlementYears() {
    return settlementYears;
  }


  public enum SideEnum {
    /** Pay the reporting currency, receive the foreign notional at settlement. */
    BUY_FOREIGN,
    /** Deliver the foreign notional, receive the reporting currency at settlement. */
    SELL_FOREIGN
  }

  private FxForward(String id, SideEnum side, double foreignNotional, double strike, double settlementYears) {
    if (!(foreignNotional > 0.0) || !(strike > 0.0) || !(settlementYears > 0.0)) {
      throw new IllegalArgumentException("need foreignNotional>0, strike>0, settlementYears>0");
    }
    this.id = id;
    this.side = side;
    this.foreignNotional = foreignNotional;
    this.strike = strike;
    this.settlementYears = settlementYears;
  }

  @Override
  public String toString() {
    return String.format(Locale.ROOT, "%-12s %-12s %,.0fm foreign @ %.4f  %.1fy",
        id, side, foreignNotional / 1e6, strike, settlementYears);
  }

  @Override
  public double grossNotional() {
    return foreignNotional * strike;
  }

  @Override
  public double effectiveMaturityYears() {
    return settlementYears;
  }

  @Override
  public ADouble markToMarket(Path path, double t) {
    if (t >= settlementYears - 1.0e-9) {
      return path.recorder().constant(0.0);
    }
    ADouble domesticDiscount = path.rates().bond(path.shortRate(), t, settlementYears);
    ADouble foreignDiscount = path.foreignDiscount(t, settlementYears);
    ADouble forward = path.fxSpot().mul(foreignDiscount).div(domesticDiscount);
    ADouble value = forward.sub(strike).mul(domesticDiscount).mul(foreignNotional);
    return value.mul(side == SideEnum.BUY_FOREIGN ? 1.0 : -1.0);
  }
}
