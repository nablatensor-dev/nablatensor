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
package com.nablatensor.lattice;

import com.nablatensor.lattice.LatticePayoff.ExerciseSchedule;
import com.nablatensor.quant.OptionType;

/**
 * Greeks from a binomial tree. {@link #delta} and {@link #gamma} are read
 * straight off the first two backward-induction slices — they cost nothing
 * beyond the price. {@link #vega}, {@link #rho} and {@link #theta} rebuild the
 * tree with a bumped parameter (two or three extra trees), which for an
 * {@code O(n^2)} lattice is still cheap.
 *
 * <p>This is the tree engine's answer to the adjoint sweep: there is no tape, so
 * the sensitivities come from the lattice geometry and small rebuilds instead.
 */
public record LatticeGreeks(double price, double delta, double gamma,
                            double vega, double rho, double theta) {

  /**
   * @param spot     spot
   * @param rate     risk-free rate
   * @param dividend continuous dividend yield
   * @param vol      volatility
   * @param maturity years to expiry
   * @param steps    tree steps
   * @param method   lattice parameterisation
   * @param type     call or put
   * @param strike   strike
   * @param schedule European / American / Bermudan exercise
   */
  public static LatticeGreeks vanilla(double spot, double rate, double dividend, double vol,
                                      double maturity, int steps, BinomialTree.Method method,
                                      OptionType type, double strike, ExerciseSchedule schedule) {
    BinomialTree tree = BinomialTree.of(spot, rate, dividend, vol, maturity, steps, method);
    double price = tree.priceVanilla(type, strike, schedule);

    double[] v1 = tree.slice1();
    double[] v2 = tree.slice2();
    double[] s1 = tree.nodeSpots1();
    double[] s2 = tree.nodeSpots2();

    double delta = (v1[1] - v1[0]) / (s1[1] - s1[0]);
    double deltaUp = (v2[2] - v2[1]) / (s2[2] - s2[1]);
    double deltaDn = (v2[1] - v2[0]) / (s2[1] - s2[0]);
    double gamma = (deltaUp - deltaDn) / (0.5 * (s2[2] - s2[0]));

    double hv = 1e-4 * Math.max(1.0, vol);
    double vegaUp = price(spot, rate, dividend, vol + hv, maturity, steps, method, type, strike, schedule);
    double vegaDn = price(spot, rate, dividend, vol - hv, maturity, steps, method, type, strike, schedule);
    double vega = (vegaUp - vegaDn) / (2 * hv);

    double hr = 1e-4 * Math.max(1.0, Math.abs(rate));
    double rhoUp = price(spot, rate + hr, dividend, vol, maturity, steps, method, type, strike, schedule);
    double rhoDn = price(spot, rate - hr, dividend, vol, maturity, steps, method, type, strike, schedule);
    double rho = (rhoUp - rhoDn) / (2 * hr);

    // theta = dV/dT ; the tree's own 2-step slice gives a calendar-time estimate too,
    // but a small T bump keeps it consistent with the reported price.
    double ht = 1e-4 * maturity;
    double thetaUp = price(spot, rate, dividend, vol, maturity + ht, steps, method, type, strike, schedule);
    double thetaDn = price(spot, rate, dividend, vol, maturity - ht, steps, method, type, strike, schedule);
    double theta = (thetaUp - thetaDn) / (2 * ht);

    return new LatticeGreeks(price, delta, gamma, vega, rho, theta);
  }

  private static double price(double spot, double rate, double dividend, double vol, double maturity,
                              int steps, BinomialTree.Method method, OptionType type, double strike,
                              ExerciseSchedule schedule) {
    return BinomialTree.of(spot, rate, dividend, vol, maturity, steps, method)
        .priceVanilla(type, strike, schedule);
  }
}
