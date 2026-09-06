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

import com.nablatensor.quant.OptionType;

/**
 * The exercise value of a contract at a lattice node, as a function of the
 * underlying at that node. Used both at expiry and — for an American or
 * Bermudan contract — at every earlier exercisable step.
 */
@FunctionalInterface
public interface LatticePayoff {

  double exerciseValue(double underlying);

  /** {@code max(sign * (S - K), 0)} for a vanilla call or put. */
  static LatticePayoff vanilla(OptionType type, double strike) {
    double sign = type.sign();
    return s -> Math.max(sign * (s - strike), 0.0);
  }

  /** Whether this step is an exercise opportunity (for a Bermudan schedule). */
  @FunctionalInterface
  interface ExerciseSchedule {
    boolean exercisableAtStep(int step, int totalSteps);

    ExerciseSchedule EUROPEAN = (step, n) -> step == n;
    ExerciseSchedule AMERICAN = (step, n) -> true;

    /** Exercisable on the last step and every {@code period}-th step before it. */
    static ExerciseSchedule everyNthStep(int period) {
      return (step, n) -> step == n || (step > 0 && step % period == 0);
    }
  }
}
