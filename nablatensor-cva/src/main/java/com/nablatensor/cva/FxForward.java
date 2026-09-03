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

import com.nablatensor.engine.SDouble;
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
public record FxForward(String id, Side side, double foreignNotional, double strike, double settlementYears)
    implements CvaTrade {

  public enum Side { BUY_FOREIGN, SELL_FOREIGN }

  public FxForward {
    if (!(foreignNotional > 0.0) || !(strike > 0.0) || !(settlementYears > 0.0)) {
      throw new IllegalArgumentException("need foreignNotional>0, strike>0, settlementYears>0");
    }
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
  public SDouble markToMarket(Path path, double t) {
    if (t >= settlementYears - 1.0e-9) {
      return path.recorder().constant(0.0);
    }
    SDouble domesticDiscount = path.rates().bond(path.shortRate(), t, settlementYears);
    SDouble foreignDiscount = path.foreignDiscount(t, settlementYears);
    SDouble forward = path.fxSpot().mul(foreignDiscount).div(domesticDiscount);
    SDouble value = forward.sub(strike).mul(domesticDiscount).mul(foreignNotional);
    return value.mul(side == Side.BUY_FOREIGN ? 1.0 : -1.0);
  }
}
