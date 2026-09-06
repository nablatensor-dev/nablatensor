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

import com.nablatensor.quant.analytic.AnalyticGreeks;
import com.nablatensor.quant.analytic.Bachelier;
import java.util.ArrayList;
import java.util.List;

/**
 * Fits the two Hull-White parameters {@code (a, sigma)} to a grid of European
 * swaption quotes, given today's discount curve.
 *
 * <p>Each quote is turned into a target price with {@link Bachelier} (the market
 * quotes an ATM normal vol on the forward swap rate); the model price is the
 * {@link HullWhiteAnalytic#payerSwaption Jamshidian swaption}. The objective is
 * the sum of squared price residuals, minimised by a bounded Nelder-Mead search
 * — the analytic swaption is not recordable (a root-find and {@code N(x)}), so
 * this is the numerical rather than the adjoint calibration route; the adjoint
 * route runs against the {@link HullWhite1F} Monte-Carlo swaption instead.
 */
public final class HullWhiteCalibration {

  /** One ATM European swaption quote: an option expiry, a swap tenor, and a normal vol. */
  public record SwaptionQuote(double expiryYears, int swapTenorYears, double accrual, double normalVol) {

    public int periods() {
      return (int) Math.round(swapTenorYears / accrual);
    }
  }

  /** The fitted parameters and the fit quality. */
  public record Result(double a, double sigma, double rmsePrice, int iterations, boolean converged,
                       double[] modelPrices, double[] targetPrices) {

    public HullWhiteAnalytic model(YieldCurve discountCurve) {
      return HullWhiteAnalytic.of(discountCurve, a, sigma);
    }
  }

  private HullWhiteCalibration() {
  }

  public static Result calibrate(YieldCurve discountCurve, List<SwaptionQuote> quotes,
                                 double aInit, double sigmaInit) {
    int m = quotes.size();
    double[] target = new double[m];
    double[] forward = new double[m];
    double[] annuity = new double[m];
    for (int k = 0; k < m; k++) {
      SwaptionQuote q = quotes.get(k);
      double[] payTimes = new double[q.periods()];
      double[] tau = new double[q.periods()];
      for (int i = 0; i < q.periods(); i++) {
        payTimes[i] = q.expiryYears() + (i + 1) * q.accrual();
        tau[i] = q.accrual();
      }
      double ann = 0.0;
      for (int i = 0; i < payTimes.length; i++) {
        ann += tau[i] * discountCurve.discountFactor(payTimes[i]);
      }
      double fwd = (discountCurve.discountFactor(q.expiryYears())
          - discountCurve.discountFactor(payTimes[payTimes.length - 1])) / ann;
      annuity[k] = ann;
      forward[k] = fwd;
      // ATM payer swaption price from the quoted normal vol (undiscounted Bachelier * annuity).
      AnalyticGreeks bach = Bachelier.of(OptionType.CALL, fwd, fwd, q.expiryYears(), 0.0, q.normalVol());
      target[k] = bach.price() * ann;
    }

    java.util.function.DoubleBinaryOperator sse = (a, sigma) -> {
      if (a <= 0.0 || sigma <= 0.0) {
        return 1e18;
      }
      HullWhiteAnalytic hw = HullWhiteAnalytic.of(discountCurve, a, sigma);
      double s = 0.0;
      for (int k = 0; k < m; k++) {
        SwaptionQuote q = quotes.get(k);
        double model = hw.payerSwaption(q.expiryYears(), q.accrual(), q.periods(), forward[k]);
        double d = model - target[k];
        s += d * d;
      }
      return s;
    };

    NelderMead.Result nm = NelderMead.minimise(sse, aInit, sigmaInit, 0.03, 0.003, 400, 1e-16);

    HullWhiteAnalytic hw = HullWhiteAnalytic.of(discountCurve, nm.x0(), nm.x1());
    double[] modelPrices = new double[m];
    double sumSq = 0.0;
    for (int k = 0; k < m; k++) {
      SwaptionQuote q = quotes.get(k);
      modelPrices[k] = hw.payerSwaption(q.expiryYears(), q.accrual(), q.periods(), forward[k]);
      double d = modelPrices[k] - target[k];
      sumSq += d * d;
    }
    return new Result(nm.x0(), nm.x1(), Math.sqrt(sumSq / m), nm.iterations(), nm.converged(),
        modelPrices, target);
  }

  /** Convenience: an equal-accrual co-terminal / diagonal grid. */
  public static List<SwaptionQuote> grid(double[] expiries, int[] tenors, double accrual, double[] normalVols) {
    if (expiries.length != tenors.length || expiries.length != normalVols.length) {
      throw new IllegalArgumentException("expiries, tenors and normalVols must be the same length");
    }
    List<SwaptionQuote> q = new ArrayList<>();
    for (int i = 0; i < expiries.length; i++) {
      q.add(new SwaptionQuote(expiries[i], tenors[i], accrual, normalVols[i]));
    }
    return q;
  }

  // ---- a small bounded Nelder-Mead ----------------------------------

  static final class NelderMead {

    record Result(double x0, double x1, double value, int iterations, boolean converged) {}

    static Result minimise(java.util.function.DoubleBinaryOperator f, double s0, double s1,
                           double step0, double step1, int maxIter, double tol) {
      double[][] p = {
          {s0, s1},
          {s0 + step0, s1},
          {s0, s1 + step1},
      };
      double[] fv = new double[3];
      for (int i = 0; i < 3; i++) {
        fv[i] = f.applyAsDouble(p[i][0], p[i][1]);
      }

      int iter = 0;
      for (; iter < maxIter; iter++) {
        // order: best (0), ..., worst (2)
        for (int i = 0; i < 3; i++) {
          for (int j = i + 1; j < 3; j++) {
            if (fv[j] < fv[i]) {
              swap(p, fv, i, j);
            }
          }
        }
        if (Math.abs(fv[2] - fv[0]) <= tol * (Math.abs(fv[0]) + tol)) {
          break;
        }
        double cx = 0.5 * (p[0][0] + p[1][0]);
        double cy = 0.5 * (p[0][1] + p[1][1]);
        // reflection
        double rx = cx + (cx - p[2][0]);
        double ry = cy + (cy - p[2][1]);
        double fr = f.applyAsDouble(rx, ry);
        if (fr < fv[0]) {
          double ex = cx + 2.0 * (cx - p[2][0]);
          double ey = cy + 2.0 * (cy - p[2][1]);
          double fe = f.applyAsDouble(ex, ey);
          if (fe < fr) {
            p[2] = new double[] {ex, ey};
            fv[2] = fe;
          } else {
            p[2] = new double[] {rx, ry};
            fv[2] = fr;
          }
        } else if (fr < fv[1]) {
          p[2] = new double[] {rx, ry};
          fv[2] = fr;
        } else {
          double gx = cx + 0.5 * (p[2][0] - cx);
          double gy = cy + 0.5 * (p[2][1] - cy);
          double fg = f.applyAsDouble(gx, gy);
          if (fg < fv[2]) {
            p[2] = new double[] {gx, gy};
            fv[2] = fg;
          } else {
            for (int i = 1; i < 3; i++) {
              p[i][0] = p[0][0] + 0.5 * (p[i][0] - p[0][0]);
              p[i][1] = p[0][1] + 0.5 * (p[i][1] - p[0][1]);
              fv[i] = f.applyAsDouble(p[i][0], p[i][1]);
            }
          }
        }
      }
      for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 3; j++) {
          if (fv[j] < fv[i]) {
            swap(p, fv, i, j);
          }
        }
      }
      return new Result(p[0][0], p[0][1], fv[0], iter, iter < maxIter);
    }

    private static void swap(double[][] p, double[] fv, int i, int j) {
      double[] tp = p[i];
      p[i] = p[j];
      p[j] = tp;
      double tf = fv[i];
      fv[i] = fv[j];
      fv[j] = tf;
    }
  }
}
