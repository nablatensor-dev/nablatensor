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
package com.nablatensor.quant.transform;

import com.nablatensor.quant.OptionType;
import java.util.List;

/**
 * Fits the five Heston parameters {@code (v0, kappa, theta, xi, rho)} to a grid
 * of European option quotes by minimising the sum of squared price residuals,
 * every model price coming from {@link CosMethod}. Deterministic and fast — a
 * full surface fit is a fraction of a second — which is why the transform route
 * is the calibration counterpart to the Monte-Carlo models.
 */
public final class HestonCosCalibrator {

  /** One quote: strike, maturity, and the observed option price (a call). */
  public record Quote(double strike, double maturity, double price) {}

  /** Fitted parameters and the fit quality. */
  public record Result(double v0, double kappa, double theta, double xi, double rho,
                       double rmse, int iterations, boolean converged) {

    public HestonCf cf(double rate) {
      return new HestonCf(rate, v0, kappa, theta, xi, rho);
    }
  }

  private HestonCosCalibrator() {
  }

  public static Result calibrate(double spot, double rate, List<Quote> quotes,
                                 double[] start) {
    // params: v0, kappa, theta, xi, rho
    double[] lo = {1e-4, 1e-2, 1e-4, 1e-3, -0.999};
    double[] hi = {1.0, 20.0, 1.0, 5.0, 0.999};
    double[] step = {0.01, 0.5, 0.01, 0.05, 0.05};

    java.util.function.ToDoubleFunction<double[]> sse = x -> {
      HestonCf cf = new HestonCf(rate, x[0], x[1], x[2], x[3], x[4]);
      double s = 0.0;
      for (Quote q : quotes) {
        double model = CosMethod.price(cf, OptionType.CALL, spot, q.strike(), rate, q.maturity());
        double d = model - q.price();
        s += d * d;
      }
      return s;
    };

    NelderMead.Result nm = NelderMead.minimise(sse, start, step, lo, hi, 2000, 1e-16);
    double[] x = nm.x();
    double rmse = Math.sqrt(sse.applyAsDouble(x) / quotes.size());
    return new Result(x[0], x[1], x[2], x[3], x[4], rmse, nm.iterations(), nm.converged());
  }

  /** A reasonable cold start for equity-index data. */
  public static double[] defaultStart(double atmVariance) {
    return new double[] {atmVariance, 2.0, atmVariance, 0.5, -0.6};
  }
}
