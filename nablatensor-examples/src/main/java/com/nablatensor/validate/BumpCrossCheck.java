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
package com.nablatensor.validate;

import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Product;
import java.util.function.BiFunction;
import java.util.function.ToDoubleFunction;

/**
 * Central bump-and-revalue on the scalar oracle, run with common random numbers
 * (the same seed on every leg) so the difference against the adjoint gradient is
 * the discretisation error of the bump, not Monte-Carlo noise.
 */
public record BumpCrossCheck(EquityMarket adjoint, EquityMarket bump, EquityMarket absDiff,
                             double relativeBump) {

  /** @param relativeBump bump size as a fraction of each input's magnitude, e.g. {@code 0.005} */
  static BumpCrossCheck run(Product<EquityMarket> product, EquityMarket market, int steps,
                            boolean fp32, long scenarios, long seed, double relativeBump,
                            EquityMarket adjointGreeks) {
    try (MonteCarlo<EquityMarket> price = configure(MonteCarlo.of(product)
        .market(market).steps(steps).priceOnly().on("cpu"), fp32).build()) {

      double[] grad = new double[EquityFactorEnum.values().length];
      for (EquityFactorEnum factor : EquityFactorEnum.values()) {
        double h = relativeBump * Math.max(1.0, Math.abs(factor.value.applyAsDouble(market)));
        grad[factor.ordinal()] = (price.run(factor.bump.apply(market, h), scenarios, seed).price()
            - price.run(factor.bump.apply(market, -h), scenarios, seed).price()) / (2 * h);
      }
      EquityMarket bumpGreeks = EquityMarket.of().spot(grad[0]).strike(grad[1]).vol(grad[2]).rate(grad[3]).maturity(grad[4]).build();
      EquityMarket diff = EquityMarket.of().spot(Math.abs(grad[0] - adjointGreeks.spot())).strike(Math.abs(grad[1] - adjointGreeks.strike())).vol(Math.abs(grad[2] - adjointGreeks.vol())).rate(Math.abs(grad[3] - adjointGreeks.rate())).maturity(Math.abs(grad[4] - adjointGreeks.maturity())).build();
      return new BumpCrossCheck(adjointGreeks, bumpGreeks, diff, relativeBump);
    }
  }

  private static MonteCarlo.Builder<EquityMarket> configure(MonteCarlo.Builder<EquityMarket> b, boolean fp32) {
    return fp32 ? b.fp32() : b.fp64();
  }

  private enum EquityFactorEnum {
    SPOT(EquityMarket::spot, (market, bump) -> market.withSpot(market.spot() + bump)),
    STRIKE(EquityMarket::strike, (market, bump) -> market.withStrike(market.strike() + bump)),
    VOL(EquityMarket::vol, (market, bump) -> market.withVol(market.vol() + bump)),
    RATE(EquityMarket::rate, (market, bump) -> market.withRate(market.rate() + bump)),
    MATURITY(EquityMarket::maturity, (market, bump) -> market.withMaturity(market.maturity() + bump));

    private final ToDoubleFunction<EquityMarket> value;
    private final BiFunction<EquityMarket, Double, EquityMarket> bump;

    EquityFactorEnum(ToDoubleFunction<EquityMarket> value,
                    BiFunction<EquityMarket, Double, EquityMarket> bump) {
      this.value = value;
      this.bump = bump;
    }
  }

  public double maxAbsDiff() {
    return Math.max(Math.max(absDiff.spot(), absDiff.strike()),
        Math.max(absDiff.vol(), Math.max(absDiff.rate(), absDiff.maturity())));
  }
}
