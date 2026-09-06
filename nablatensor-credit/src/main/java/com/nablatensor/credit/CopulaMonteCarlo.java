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
package com.nablatensor.credit;

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.Nabla;
import com.nablatensor.engine.SDouble;
import com.nablatensor.ops.SpecialFn;
import com.nablatensor.ops.Smooth;
import java.util.function.BiConsumer;

/**
 * The one-factor Gaussian copula as a <em>recorded</em> Monte-Carlo, so a
 * tranche's correlation delta and its sensitivity to the pool default
 * probability come from a single adjoint sweep — where the
 * {@link PortfolioLossDistribution} recursion gives the price, this gives the
 * risk.
 *
 * <p>Each path draws one systemic normal shared by every name and one
 * idiosyncratic normal per name; a name defaults when
 * {@code Phi(sqrt(rho) M + sqrt(1 - rho) Z_i) < pd}, monitored with a smoothed
 * indicator so the payoff stays differentiable. The output is the discounted
 * tranche loss to a single horizon (a protection-leg-only PV on unit tranche
 * notional).
 */
public final class CopulaMonteCarlo {

  private CopulaMonteCarlo() {
  }

  /**
   * @param attach   tranche attachment, fraction of the pool
   * @param detach   tranche detachment
   * @param names    pool size
   * @param lgd      loss given default, fraction of one name's notional
   * @param maturity horizon in years
   * @param rate     flat discount rate
   * @param width    smoothing width of the default indicator, in probability units (e.g. {@code 5e-3})
   */
  public static BiConsumer<AadRecorder, Nabla.Inputs<CopulaMarket>> trancheLoss(
      double attach, double detach, int names, double lgd, double maturity, double rate, double width) {
    if (!(detach > attach)) {
      throw new IllegalArgumentException("need detach > attach");
    }
    double trancheWidth = detach - attach;
    double discount = Math.exp(-rate * maturity);
    return (rec, in) -> {
      SDouble rho = in.of(CopulaMarket::rho);
      SDouble pd = in.of(CopulaMarket::pd);
      SDouble sqrtRho = rho.sqrt();
      SDouble sqrtComp = rec.constant(1.0).sub(rho).sqrt();
      SDouble m = rec.randn();

      SDouble defaults = rec.constant(0.0);
      for (int i = 0; i < names; i++) {
        SDouble xi = sqrtRho.mul(m).add(sqrtComp.mul(rec.randn()));
        SDouble u = SpecialFn.normCdf(rec, xi);
        defaults = defaults.add(Smooth.lt(rec, u, pd, width));
      }

      SDouble portfolioLoss = defaults.mul(lgd / names);
      SDouble trancheLoss = portfolioLoss.sub(attach).max(0.0).min(trancheWidth);
      rec.output(trancheLoss.mul(discount));
    };
  }
}
