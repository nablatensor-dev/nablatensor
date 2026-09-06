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
import java.util.function.IntToDoubleFunction;

/**
 * Price versus step count, with a Richardson extrapolation of the sequence — the
 * pedagogical artifact for "the tree converges to the closed form". For a
 * Cox-Ross-Rubinstein European option the error is {@code O(1/n)} with an
 * even/odd oscillation, so averaging {@code n} and {@code n+1} (a one-step
 * Richardson) removes the leading term and leaves {@code O(1/n^2)}.
 */
public record ConvergenceTable(int[] steps, double[] prices, double richardsonExtrapolated) {

  public static ConvergenceTable of(IntToDoubleFunction priceAtSteps, int[] steps) {
    double[] prices = new double[steps.length];
    for (int i = 0; i < steps.length; i++) {
      prices[i] = priceAtSteps.applyAsDouble(steps[i]);
    }
    // one-step Richardson on the two largest step counts (n and its neighbour)
    int last = steps.length - 1;
    double rich = prices[last];
    if (steps.length >= 2) {
      double nLast = steps[last];
      double nPrev = steps[last - 1];
      rich = (nLast * prices[last] - nPrev * prices[last - 1]) / (nLast - nPrev);
    }
    return new ConvergenceTable(steps.clone(), prices, rich);
  }

  /** Convenience for a vanilla option on a CRR tree. */
  public static ConvergenceTable crrVanilla(double spot, double rate, double dividend, double vol,
                                            double maturity, OptionType type, double strike,
                                            ExerciseSchedule schedule, int[] steps) {
    return of(n -> BinomialTree.of(spot, rate, dividend, vol, maturity, n, BinomialTree.Method.CRR)
        .priceVanilla(type, strike, schedule), steps);
  }
}
