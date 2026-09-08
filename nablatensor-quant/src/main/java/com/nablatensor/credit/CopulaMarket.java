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

/**
 * A homogeneous credit pool for the recorded copula Monte-Carlo: one systemic
 * correlation and one horizon default probability shared by every name. Both are
 * differentiable inputs, so one adjoint sweep of a tranche payoff returns its
 * correlation delta and its sensitivity to the pool default probability.
 *
 * @param rho correlation to the systemic factor, in {@code (0, 1)}
 * @param pd  default probability of a name to the horizon, in {@code (0, 1)}
 */
public record CopulaMarket(double rho, double pd) {

  public CopulaMarket validated() {
    if (!(rho > 0 && rho < 1 && pd > 0 && pd < 1)) {
      throw new IllegalArgumentException("need rho, pd in (0, 1): " + this);
    }
    return this;
  }

  public static CopulaMarket base() {
    return new CopulaMarket(0.30, 0.05);
  }
}
