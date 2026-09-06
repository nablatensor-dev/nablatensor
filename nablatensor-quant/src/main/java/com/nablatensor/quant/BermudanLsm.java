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

import com.nablatensor.engine.SDouble;
import com.nablatensor.ops.Smooth;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Least-squares Monte-Carlo pricing of a Bermudan option by <em>policy
 * optimisation</em>: the continuation value at each exercise date is a low-degree
 * polynomial in log-moneyness whose coefficients {@code beta} are chosen to
 * maximise the price under the smoothed exercise rule. The optimised price is a
 * valid lower bound (a sub-optimal policy can only lose value), and at the
 * optimum {@code d(price)/d(beta) = 0}, so by the envelope theorem the market
 * Greeks read off the same tape with {@code beta} held fixed are correct to
 * first order.
 *
 * <p>This fills in {@link BermudanOption}'s Phase-3 hook without an engine
 * "probe replay": the whole valuation stays on one recorded tape, and the
 * coefficient gradient comes from {@link MultiOutput} (one forward sweep, one
 * reverse sweep per output) exactly like every other adjoint calibration.
 */
public final class BermudanLsm {

  /** The optimised Bermudan price, its Greeks, and the fitted exercise boundary. */
  public record Result(double price, double europeanFloor, double standardError,
                       EquityMarket greeks, double[][] boundaryCoefficients,
                       int iterations, boolean converged) {

    /** Early-exercise premium over the European. */
    public double earlyExercisePremium() {
      return price - europeanFloor;
    }
  }

  private BermudanLsm() {
  }

  /**
   * @param market        the equity market
   * @param type          call or put on the spot
   * @param exerciseDates equally spaced exercise opportunities (the last is expiry)
   * @param stepsPerDate  GBM sub-steps between consecutive exercise dates
   * @param polyDegree    degree of the log-moneyness polynomial for the continuation value
   * @param decisionWidth smoothing width of the exercise decision, in spot units
   * @param scenarios     Monte-Carlo paths
   * @param seed          RNG seed
   */
  public static Result price(EquityMarket market, OptionType type, int exerciseDates, int stepsPerDate,
                             int polyDegree, double decisionWidth, long scenarios, long seed) {
    int perDate = polyDegree + 1;
    int decisionDates = exerciseDates - 1;              // the last date is forced exercise
    int nBeta = decisionDates * perDate;
    int totalSteps = exerciseDates * stepsPerDate;
    double sRef = market.spot();

    MultiOutput.Measures measures = rec -> {
      SDouble s0 = rec.input("S0", market.spot());
      SDouble k = rec.input("K", market.strike());
      SDouble vol = rec.input("sigma", market.vol());
      SDouble r = rec.input("r", market.rate());
      SDouble tt = rec.input("T", market.maturity());
      GbmPath model = new GbmPath(rec, r, vol, totalSteps, tt);
      SDouble stepDisc = r.neg().mul(tt).div(totalSteps).exp();

      SDouble discount = rec.constant(1.0);
      SDouble alive = rec.constant(1.0);
      SDouble value = rec.constant(0.0);
      SDouble euro = rec.constant(0.0);
      SDouble s = s0;
      int stepIdx = 0;
      for (int d = 0; d < exerciseDates; d++) {
        for (int step = 0; step < stepsPerDate; step++) {
          s = model.step(s, rec.randn(), stepIdx++);
          discount = discount.mul(stepDisc);
        }
        SDouble exercise = (type == OptionType.CALL ? s.sub(k) : k.sub(s)).max(0.0);
        boolean last = d == exerciseDates - 1;
        SDouble exerciseNow;
        if (last) {
          exerciseNow = alive;
        } else {
          SDouble x = s.div(sRef).log();
          SDouble contEst = rec.constant(0.0);
          SDouble xp = rec.constant(1.0);
          for (int j = 0; j < perDate; j++) {
            contEst = contEst.add(rec.input("cv:" + d + ":" + j, 0.0).mul(xp));
            xp = xp.mul(x);
          }
          exerciseNow = alive.mul(Smooth.gt(rec, exercise.sub(contEst), 0.0, decisionWidth));
        }
        value = value.add(exerciseNow.mul(exercise).mul(discount));
        alive = alive.sub(exerciseNow);
        if (last) {
          euro = exercise.mul(discount);
        }
      }
      Map<String, SDouble> out = new LinkedHashMap<>();
      out.put("price", value);
      out.put("european", euro);
      return out;
    };

    try (MultiOutput mo = MultiOutput.of(measures).on("cpu-jit").build()) {
      double[] beta = new double[nBeta];                 // start: exercise whenever in the money
      Map<String, Double> over = betaMap(beta, decisionDates, perDate);
      MultiOutput.Result r0 = mo.run(over, scenarios, seed);
      double best = r0.value("price");

      // Gradient ascent on the price w.r.t. beta, with a backtracking step.
      double step = 0.5 * Math.max(1.0, market.strike());
      int iter = 0;
      boolean converged = false;
      for (; iter < 40; iter++) {
        Map<String, Double> g = mo.run(betaMap(beta, decisionDates, perDate), scenarios, seed).gradient("price");
        double[] grad = new double[nBeta];
        double gnorm = 0.0;
        int idx = 0;
        for (int d = 0; d < decisionDates; d++) {
          for (int j = 0; j < perDate; j++) {
            grad[idx] = g.getOrDefault("cv:" + d + ":" + j, 0.0);
            gnorm += grad[idx] * grad[idx];
            idx++;
          }
        }
        gnorm = Math.sqrt(gnorm);
        if (gnorm < 1e-9 || step < 1e-6) {
          converged = true;
          break;
        }
        boolean improved = false;
        for (int ls = 0; ls < 12 && !improved; ls++) {
          double[] trial = beta.clone();
          for (int i = 0; i < nBeta; i++) {
            trial[i] += step * grad[i] / gnorm;
          }
          double trialPrice = mo.run(betaMap(trial, decisionDates, perDate), scenarios, seed).value("price");
          if (trialPrice > best + 1e-10) {
            beta = trial;
            best = trialPrice;
            improved = true;
            step *= 1.3;
          } else {
            step *= 0.5;
          }
        }
        if (!improved) {
          converged = true;
          break;
        }
      }

      MultiOutput.Result fin = mo.run(betaMap(beta, decisionDates, perDate), scenarios, seed);
      Map<String, Double> g = fin.gradient("price");
      EquityMarket greeks = new EquityMarket(
          g.getOrDefault("S0", 0.0), g.getOrDefault("K", 0.0), g.getOrDefault("sigma", 0.0),
          g.getOrDefault("r", 0.0), g.getOrDefault("T", 0.0));

      double[][] boundary = new double[decisionDates][perDate];
      int i = 0;
      for (int d = 0; d < decisionDates; d++) {
        for (int j = 0; j < perDate; j++) {
          boundary[d][j] = beta[i++];
        }
      }
      return new Result(fin.value("price"), fin.value("european"), fin.standardError("price"),
          greeks, boundary, iter, converged);
    }
  }

  private static Map<String, Double> betaMap(double[] beta, int dates, int perDate) {
    Map<String, Double> m = new LinkedHashMap<>();
    int i = 0;
    for (int d = 0; d < dates; d++) {
      for (int j = 0; j < perDate; j++) {
        m.put("cv:" + d + ":" + j, beta[i++]);
      }
    }
    return m;
  }
}
