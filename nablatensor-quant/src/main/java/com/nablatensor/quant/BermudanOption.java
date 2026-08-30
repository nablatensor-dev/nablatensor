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
package com.nablatensor.quant;

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.SDouble;
import com.nablatensor.engine.Nabla;
import com.nablatensor.ops.Smooth;

/**
 * Bermudan option <em>shell</em>: the exercise-schedule machinery, with a
 * pluggable {@link ContinuationValue}. The path is walked once; at each exercise
 * date a <em>smoothed</em> decision compares the immediate exercise value to the
 * estimated continuation value and moves the not-yet-exercised probability mass.
 *
 * <p>With the default {@link ContinuationValue#EUROPEAN} the continuation
 * estimate is zero, so exercise only ever happens at the last date and the
 * price collapses to the European. A least-squares Monte-Carlo continuation
 * estimator — replay to collect path states, regress on the host, replay again
 * with the fitted continuation as {@code setInput} — is Phase 3; it plugs in
 * here without touching this class.
 */
public final class BermudanOption {

  /** Estimate of the discounted continuation value at an exercise date. */
  @FunctionalInterface
  public interface ContinuationValue {
    SDouble estimate(AadRecorder rec, int dateIndex, SDouble spot, SDouble discountFactor);

    /** Continuation always dominates immediate exercise, so exercise happens only
     *  at the last date: the Bermudan collapses to its European. */
    ContinuationValue EUROPEAN = (rec, i, spot, df) -> rec.constant(1e18);

    /** Exercise as soon as the option is in the money (a valid, generally sub-optimal rule). */
    ContinuationValue EXERCISE_WHEN_ITM = (rec, i, spot, df) -> rec.constant(0.0);
  }

  private BermudanOption() {
  }

  /**
   * @param type          call or put on the spot
   * @param exerciseDates number of equally spaced exercise opportunities (the last is expiry)
   * @param stepsPerDate  GBM sub-steps between consecutive exercise dates
   * @param decisionWidth smoothing width of the exercise decision, in spot units
   * @param continuation  continuation-value estimator ({@link ContinuationValue#EUROPEAN} for the shell)
   */
  public static Product<EquityMarket> option(OptionType type, int exerciseDates, int stepsPerDate,
                               double decisionWidth, ContinuationValue continuation) {
    return new Named("Bermudan " + type, (rec, in, ignoredGrid) -> {
      SDouble spot = in.of(EquityMarket::spot);
      SDouble strike = in.of(EquityMarket::strike);
      SDouble rate = in.of(EquityMarket::rate);
      SDouble maturity = in.of(EquityMarket::maturity);
      GbmPath model = new GbmPath(rec, rate, in.of(EquityMarket::vol), exerciseDates * stepsPerDate, maturity);

      SDouble stepDiscount = rate.neg().mul(maturity).div(exerciseDates * stepsPerDate).exp();
      SDouble discount = rec.constant(1.0);
      SDouble alive = rec.constant(1.0);
      SDouble value = rec.constant(0.0);
      SDouble s = spot;
      int stepIdx = 0;

      for (int d = 0; d < exerciseDates; d++) {
        for (int k = 0; k < stepsPerDate; k++) {
          s = model.step(s, rec.randn(), stepIdx++);
          discount = discount.mul(stepDiscount);
        }
        SDouble exercise = (type == OptionType.CALL ? s.sub(strike) : strike.sub(s)).max(0.0);
        SDouble contEst = continuation.estimate(rec, d, s, discount);
        boolean lastDate = d == exerciseDates - 1;
        SDouble exerciseNow = lastDate
            ? alive
            : alive.mul(Smooth.gt(rec, exercise.sub(contEst), 0.0, decisionWidth));
        value = value.add(exerciseNow.mul(exercise).mul(discount));
        alive = alive.sub(exerciseNow);
      }
      rec.output(value);
    });
  }

  private record Named(String label, Product<EquityMarket> body) implements Product<EquityMarket> {
    @Override
    public void record(AadRecorder rec, Nabla.Inputs<EquityMarket> in, TimeGrid grid) {
      body.record(rec, in, grid);
    }
  }
}
