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

import com.nablatensor.codegen.Of;

/**
 * A homogeneous credit pool for the recorded copula Monte-Carlo: one systemic
 * correlation and one horizon default probability shared by every name. Both are
 * differentiable inputs, so one adjoint sweep of a tranche payoff returns its
 * correlation delta and its sensitivity to the pool default probability.
 *
 * @param rho correlation to the systemic factor, in {@code (0, 1)}
 * @param pd  default probability of a name to the horizon, in {@code (0, 1)}
 */
@Of
public final class CopulaMarket {

  private CopulaMarket(double rho, double pd) {
    this.rho = rho;
    this.pd = pd;
  }

  private final double rho;
  private final double pd;
  static CopulaMarket create(double rho, double pd) {
    return new CopulaMarket(rho, pd);
  }

  public static CopulaMarketBuilder of() {
    return new CopulaMarketBuilder();
  }

  public double rho() {
    return rho;
  }

  public double pd() {
    return pd;
  }


  public CopulaMarket validated() {
    if (!(rho > 0 && rho < 1 && pd > 0 && pd < 1)) {
      throw new IllegalArgumentException("need rho, pd in (0, 1): " + this);
    }
    return this;
  }

  public static CopulaMarket base() {
    return CopulaMarket.of().rho(0.30).pd(0.05).build();
  }
}
