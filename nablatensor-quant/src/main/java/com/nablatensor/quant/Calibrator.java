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
import com.nablatensor.engine.Nabla;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Least-squares calibration over a <em>recorded</em> objective.
 *
 * <p>You supply a recording that reads the parameters by name
 * ({@code rec.input("alpha", ...)}), builds the sum of squared residuals against
 * the market, and calls {@code rec.output(sumSq)} once. The objective is
 * compiled to one kernel; every iteration is a {@code setInput} + one adjoint
 * sweep, so the gradient is exact and costs one extra sweep regardless of the
 * parameter count. A box-projected L-BFGS drives it.
 *
 * <pre>{@code
 * Calibrator.Result r = Calibrator.of(rec -> {
 *         SDouble alpha = rec.input("alpha", 0.2);
 *         SDouble rho   = rec.input("rho",  0.0);
 *         SDouble nu    = rec.input("nu",   0.3);
 *         SDouble beta  = rec.constant(0.5);
 *         SDouble sse = rec.constant(0.0);
 *         for (Quote q : quotes) {
 *           SDouble model = SabrHagan.blackVol(rec, alpha, beta, rho, nu, F, q.strike(), T);
 *           SDouble d = model.sub(q.vol());
 *           sse = sse.add(d.mul(d));
 *         }
 *         rec.output(sse);
 *       })
 *       .parameter("alpha", 0.2, 1e-4, 2.0)
 *       .parameter("rho",   0.0, -0.999, 0.999)
 *       .parameter("nu",    0.3, 1e-4, 5.0)
 *       .solve();
 * }</pre>
 */
public final class Calibrator {

  /** Outcome of a calibration. */
  public record Result(Map<String, Double> parameters, double objective, int iterations,
                       boolean converged, double gradientNorm) {}

  private record Param(String name, double initial, double lo, double hi) {}

  private final Consumer<AadRecorder> objective;      // scalar SSE -> L-BFGS
  private final MultiOutput.Measures residuals;        // residual vector -> Levenberg-Marquardt
  private final List<Param> params = new ArrayList<>();
  private int maxIterations = 100;
  private double tolerance = 1e-10;
  private int history = 7;
  private String engine = "cpu-jit";
  private long scenarios = 1L;                         // > 1 for a Monte-Carlo objective
  private long seed = 1L;

  private Calibrator(Consumer<AadRecorder> objective, MultiOutput.Measures residuals) {
    this.objective = objective;
    this.residuals = residuals;
  }

  /** Minimise a recorded scalar objective (usually a sum of squared residuals) with L-BFGS. */
  public static Calibrator of(Consumer<AadRecorder> objective) {
    return new Calibrator(objective, null);
  }

  /**
   * Fit a recorded <em>residual vector</em> with Levenberg-Marquardt. The body
   * records the parameters by name and returns the named residuals
   * {@code model_i - market_i}; the Jacobian each iteration comes from one
   * {@link MultiOutput} evaluation ({@code 1 + N} adjoint sweeps).
   */
  public static Calibrator leastSquares(MultiOutput.Measures residuals) {
    return new Calibrator(null, residuals);
  }

  /** Scenario count for a Monte-Carlo objective; the default {@code 1} is a deterministic one. */
  public Calibrator scenarios(long n) {
    this.scenarios = n;
    return this;
  }

  public Calibrator seed(long s) {
    this.seed = s;
    return this;
  }

  public Calibrator parameter(String name, double initial, double lo, double hi) {
    params.add(new Param(name, initial, lo, hi));
    return this;
  }

  public Calibrator maxIterations(int n) {
    this.maxIterations = n;
    return this;
  }

  public Calibrator tolerance(double t) {
    this.tolerance = t;
    return this;
  }

  public Calibrator on(String engine) {
    this.engine = engine;
    return this;
  }

  public Result solve() {
    if (params.isEmpty()) {
      throw new IllegalStateException("no parameters declared");
    }
    int n = params.size();
    double[] lo = new double[n];
    double[] hi = new double[n];
    double[] x = new double[n];
    for (int i = 0; i < n; i++) {
      lo[i] = params.get(i).lo();
      hi[i] = params.get(i).hi();
      x[i] = clamp(params.get(i).initial(), lo[i], hi[i]);
    }
    return residuals != null ? solveLm(x, lo, hi) : solveLbfgs(x, lo, hi);
  }

  // ---- Levenberg-Marquardt over a recorded residual vector -----------------

  private Result solveLm(double[] x, double[] lo, double[] hi) {
    int n = params.size();
    try (MultiOutput mo = MultiOutput.of(residuals).on(engine).build()) {
      List<String> res = mo.outputNames();
      int m = res.size();
      double lambda = 1e-3;
      double[] r = new double[m];
      double[][] j = new double[m][n];
      double cost = evalResiduals(mo, x, r, j);
      int iter = 0;
      double gInf = jTrInfNorm(j, r);

      for (; iter < maxIterations && gInf > tolerance && cost > tolerance * tolerance; iter++) {
        double[][] jtj = new double[n][n];
        double[] jtr = new double[n];
        for (int a = 0; a < n; a++) {
          for (int b = 0; b < n; b++) {
            double s = 0;
            for (int k = 0; k < m; k++) {
              s += j[k][a] * j[k][b];
            }
            jtj[a][b] = s;
          }
          double s = 0;
          for (int k = 0; k < m; k++) {
            s += j[k][a] * r[k];
          }
          jtr[a] = s;
        }

        boolean stepTaken = false;
        for (int tries = 0; tries < 12 && !stepTaken; tries++) {
          double[][] aug = new double[n][n];
          for (int a = 0; a < n; a++) {
            System.arraycopy(jtj[a], 0, aug[a], 0, n);
            aug[a][a] += lambda * (jtj[a][a] + 1e-12);
          }
          double[] delta = solveSpd(aug, jtr);           // (JtJ + lambda diag) d = Jt r
          double[] trial = new double[n];
          for (int a = 0; a < n; a++) {
            trial[a] = clamp(x[a] - delta[a], lo[a], hi[a]);
          }
          double[] rt = new double[m];
          double[][] jt = new double[m][n];
          double trialCost = evalResiduals(mo, trial, rt, jt);
          if (trialCost < cost) {
            x = trial;
            r = rt;
            j = jt;
            cost = trialCost;
            lambda = Math.max(lambda * 0.3, 1e-12);
            stepTaken = true;
          } else {
            lambda *= 3.0;
          }
        }
        if (!stepTaken) {
          break;
        }
        gInf = jTrInfNorm(j, r);
      }

      Map<String, Double> out = new LinkedHashMap<>();
      for (int i = 0; i < n; i++) {
        out.put(params.get(i).name(), x[i]);
      }
      return new Result(out, cost, iter, gInf <= tolerance || cost <= tolerance * tolerance, gInf);
    }
  }

  /** Fills {@code r} (residual values) and {@code j} (Jacobian rows); returns 0.5*||r||^2. */
  private double evalResiduals(MultiOutput mo, double[] x, double[] r, double[][] j) {
    Map<String, Double> over = new LinkedHashMap<>();
    for (int i = 0; i < params.size(); i++) {
      over.put(params.get(i).name(), x[i]);
    }
    MultiOutput.Result mr = mo.run(over, scenarios, seed);
    List<String> res = mo.outputNames();
    double cost = 0;
    for (int k = 0; k < res.size(); k++) {
      r[k] = mr.value(res.get(k));
      cost += r[k] * r[k];
      Map<String, Double> row = mr.gradient(res.get(k));
      for (int i = 0; i < params.size(); i++) {
        j[k][i] = row.getOrDefault(params.get(i).name(), 0.0);
      }
    }
    return 0.5 * cost;
  }

  private static double jTrInfNorm(double[][] j, double[] r) {
    int n = j[0].length;
    double max = 0;
    for (int a = 0; a < n; a++) {
      double s = 0;
      for (int k = 0; k < r.length; k++) {
        s += j[k][a] * r[k];
      }
      max = Math.max(max, Math.abs(s));
    }
    return max;
  }

  /** Solves a symmetric positive-definite system by Cholesky. */
  private static double[] solveSpd(double[][] a, double[] b) {
    int n = b.length;
    double[][] l = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int jj = 0; jj <= i; jj++) {
        double s = a[i][jj];
        for (int k = 0; k < jj; k++) {
          s -= l[i][k] * l[jj][k];
        }
        l[i][jj] = i == jj ? Math.sqrt(Math.max(s, 1e-18)) : s / l[jj][jj];
      }
    }
    double[] y = new double[n];
    for (int i = 0; i < n; i++) {
      double s = b[i];
      for (int k = 0; k < i; k++) {
        s -= l[i][k] * y[k];
      }
      y[i] = s / l[i][i];
    }
    double[] xr = new double[n];
    for (int i = n - 1; i >= 0; i--) {
      double s = y[i];
      for (int k = i + 1; k < n; k++) {
        s -= l[k][i] * xr[k];
      }
      xr[i] = s / l[i][i];
    }
    return xr;
  }

  // ---- L-BFGS over a recorded scalar objective ----------------------------

  private Result solveLbfgs(double[] x, double[] lo, double[] hi) {
    int n = params.size();
    try (Nabla.Pricer pricer = Nabla.model(objective).fp64().greeks().on(engine).build()) {
      Eval e0 = evaluate(pricer, x);
      double f = e0.f;
      double[] g = e0.g;

      List<double[]> sList = new ArrayList<>();
      List<double[]> yList = new ArrayList<>();

      int iter = 0;
      double gNorm = projectedInfNorm(x, g, lo, hi);
      for (; iter < maxIterations && gNorm > tolerance; iter++) {
        double[] dir = twoLoop(g, sList, yList);
        for (int i = 0; i < n; i++) {
          dir[i] = -dir[i];
        }
        if (dot(dir, g) >= 0) {                     // not a descent direction: reset to steepest
          for (int i = 0; i < n; i++) {
            dir[i] = -g[i];
          }
        }

        double step = 1.0;
        double[] xNew = null;
        Eval eNew = null;
        for (int ls = 0; ls < 30; ls++) {
          double[] trial = new double[n];
          for (int i = 0; i < n; i++) {
            trial[i] = clamp(x[i] + step * dir[i], lo[i], hi[i]);
          }
          Eval et = evaluate(pricer, trial);
          if (et.f <= f + 1e-4 * step * dot(g, dir) || et.f < f) {
            xNew = trial;
            eNew = et;
            break;
          }
          step *= 0.5;
        }
        if (xNew == null) {                          // line search stalled
          break;
        }

        double[] s = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
          s[i] = xNew[i] - x[i];
          y[i] = eNew.g[i] - g[i];
        }
        if (dot(s, y) > 1e-12) {
          sList.add(s);
          yList.add(y);
          if (sList.size() > history) {
            sList.remove(0);
            yList.remove(0);
          }
        }
        x = xNew;
        f = eNew.f;
        g = eNew.g;
        gNorm = projectedInfNorm(x, g, lo, hi);
      }

      Map<String, Double> out = new LinkedHashMap<>();
      for (int i = 0; i < n; i++) {
        out.put(params.get(i).name(), x[i]);
      }
      return new Result(out, f, iter, gNorm <= tolerance, gNorm);
    }
  }

  private record Eval(double f, double[] g) {}

  private Eval evaluate(Nabla.Pricer pricer, double[] x) {
    Nabla.Request req = pricer.value();
    for (int i = 0; i < params.size(); i++) {
      req.with(params.get(i).name(), x[i]);
    }
    Nabla.Valuation v = req.scenarios(scenarios).seed(seed).run();
    double[] g = new double[params.size()];
    for (int i = 0; i < g.length; i++) {
      g[i] = v.greek(params.get(i).name());
    }
    return new Eval(v.price(), g);
  }

  private static double[] twoLoop(double[] g, List<double[]> s, List<double[]> y) {
    int m = s.size();
    double[] q = g.clone();
    double[] alpha = new double[m];
    double[] rho = new double[m];
    for (int k = m - 1; k >= 0; k--) {
      rho[k] = 1.0 / dot(y.get(k), s.get(k));
      alpha[k] = rho[k] * dot(s.get(k), q);
      axpy(-alpha[k], y.get(k), q);
    }
    double scale = m == 0 ? 1.0
        : dot(s.get(m - 1), y.get(m - 1)) / dot(y.get(m - 1), y.get(m - 1));
    for (int i = 0; i < q.length; i++) {
      q[i] *= scale;
    }
    for (int k = 0; k < m; k++) {
      double beta = rho[k] * dot(y.get(k), q);
      axpy(alpha[k] - beta, s.get(k), q);
    }
    return q;
  }

  private static double projectedInfNorm(double[] x, double[] g, double[] lo, double[] hi) {
    double max = 0.0;
    for (int i = 0; i < x.length; i++) {
      double pg = x[i] - clamp(x[i] - g[i], lo[i], hi[i]);   // projected-gradient proxy
      max = Math.max(max, Math.abs(pg));
    }
    return max;
  }

  private static double clamp(double v, double lo, double hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  private static double dot(double[] a, double[] b) {
    double s = 0.0;
    for (int i = 0; i < a.length; i++) {
      s += a[i] * b[i];
    }
    return s;
  }

  private static void axpy(double a, double[] x, double[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] += a * x[i];
    }
  }
}
