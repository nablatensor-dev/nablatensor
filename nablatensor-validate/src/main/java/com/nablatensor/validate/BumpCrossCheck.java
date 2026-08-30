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

      double[] base = {market.spot(), market.strike(), market.vol(), market.rate(), market.maturity()};
      double[] grad = new double[5];
      for (int i = 0; i < 5; i++) {
        double h = relativeBump * Math.max(1.0, Math.abs(base[i]));
        grad[i] = (price.run(shift(market, i, h), scenarios, seed).price()
                 - price.run(shift(market, i, -h), scenarios, seed).price()) / (2 * h);
      }
      EquityMarket bumpGreeks = new EquityMarket(grad[0], grad[1], grad[2], grad[3], grad[4]);
      EquityMarket diff = new EquityMarket(
          Math.abs(grad[0] - adjointGreeks.spot()),
          Math.abs(grad[1] - adjointGreeks.strike()),
          Math.abs(grad[2] - adjointGreeks.vol()),
          Math.abs(grad[3] - adjointGreeks.rate()),
          Math.abs(grad[4] - adjointGreeks.maturity()));
      return new BumpCrossCheck(adjointGreeks, bumpGreeks, diff, relativeBump);
    }
  }

  private static MonteCarlo.Builder<EquityMarket> configure(MonteCarlo.Builder<EquityMarket> b, boolean fp32) {
    return fp32 ? b.fp32() : b.fp64();
  }

  private static EquityMarket shift(EquityMarket m, int component, double h) {
    return switch (component) {
      case 0 -> m.withSpot(m.spot() + h);
      case 1 -> m.withStrike(m.strike() + h);
      case 2 -> m.withVol(m.vol() + h);
      case 3 -> m.withRate(m.rate() + h);
      case 4 -> m.withMaturity(m.maturity() + h);
      default -> throw new IllegalArgumentException("component " + component);
    };
  }

  public double maxAbsDiff() {
    return Math.max(Math.max(absDiff.spot(), absDiff.strike()),
        Math.max(absDiff.vol(), Math.max(absDiff.rate(), absDiff.maturity())));
  }
}
