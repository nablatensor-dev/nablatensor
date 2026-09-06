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
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Two-stage bootstrap of a {@link CurveSet}: the OIS discount curve from OIS
 * deposits and par swaps, then each tenor forecast curve from deposits and par
 * swaps whose annuity and float legs discount on the OIS curve.
 *
 * <p>The whole recursion is recorded once against {@code SDouble} quotes and
 * replayed through a {@link MultiOutput}, so alongside the curves it returns the
 * exact {@code d(zero rate) / d(quote)} Jacobian from one adjoint sweep — the
 * bucket-delta transformation a rates desk applies to turn instrument PV01s into
 * zero-rate risk. The Jacobian is block lower-triangular by construction: OIS
 * zeros depend only on OIS quotes; forecast zeros also depend on the OIS quotes
 * that move their discounting.
 *
 * <p>Stylised annual construction (see {@link CurveSet}): OIS pillars are the
 * integer years {@code 1..N} (an optional sub-year deposit is allowed but only
 * used for its own pillar, not the annual annuity); each forecast curve is a
 * 1-year deposit or 1-year swap followed by consecutive integer-year swaps.
 */
public final class MultiCurveBootstrap {

  /** OIS discount curve tenor key. */
  public static final String OIS = "OIS";

  private record Inst(String curve, boolean deposit, double maturity, double quote) {
    String label() {
      return curve + (deposit ? ":dep:" : ":swap:") + trim(maturity);
    }

    static String trim(double t) {
      return t == Math.rint(t) ? Integer.toString((int) t) : Double.toString(t);
    }
  }

  private final List<Inst> ois;
  private final Map<String, List<Inst>> forecast;

  private MultiCurveBootstrap(List<Inst> ois, Map<String, List<Inst>> forecast) {
    this.ois = ois;
    this.forecast = forecast;
  }

  public static Builder builder() {
    return new Builder();
  }

  /** The bootstrapped curves plus the zero-rate Jacobian. */
  public record Result(CurveSet curves, double[][] jacobian,
                       List<String> zeroLabels, List<String> quoteLabels) {

    /** {@code d(zeroLabels[row]) / d(quoteLabels[col])}. */
    public double sensitivity(String zeroLabel, String quoteLabel) {
      int r = zeroLabels.indexOf(zeroLabel);
      int c = quoteLabels.indexOf(quoteLabel);
      if (r < 0 || c < 0) {
        throw new IllegalArgumentException("unknown label: " + zeroLabel + " / " + quoteLabel);
      }
      return jacobian[r][c];
    }
  }

  public Result solve() {
    try (Kernel k = kernel()) {
      return k.evaluate(Map.of());
    }
  }

  /**
   * Records the bootstrap recursion once and keeps the compiled kernel open, so
   * the curves and Jacobian can be re-evaluated at shifted quotes (a bump
   * cross-check, a scenario) without re-recording. Close it when done.
   */
  public Kernel kernel() {
    return new Kernel();
  }

  /** A compiled, reusable bootstrap. */
  public final class Kernel implements AutoCloseable {
    private final List<String> quoteLabels = new ArrayList<>();
    private final List<String> zeroLabels = new ArrayList<>();
    private final MultiOutput mo;

    private Kernel() {
      for (Inst i : ois) {
        quoteLabels.add(i.label());
        zeroLabels.add("z:" + i.label());
      }
      for (var e : forecast.entrySet()) {
        for (Inst i : e.getValue()) {
          quoteLabels.add(i.label());
          zeroLabels.add("z:" + i.label());
        }
      }
      this.mo = MultiOutput.of(rec -> record(rec)).on("cpu").build();
    }

    /** Re-evaluate with {@code quoteOverrides} (label -> quote); an empty map uses the base quotes. */
    public Result evaluate(Map<String, Double> quoteOverrides) {
      MultiOutput.Result r = mo.run(quoteOverrides, 1L, 0L);
      Map<String, Double> values = new LinkedHashMap<>();
      Map<String, Map<String, Double>> grads = new LinkedHashMap<>();
      for (String z : zeroLabels) {
        values.put(z, r.value(z));
        grads.put(z, r.gradient(z));
      }
      CurveSet curves = assembleCurves(values);
      double[][] jac = new double[zeroLabels.size()][quoteLabels.size()];
      for (int rIdx = 0; rIdx < zeroLabels.size(); rIdx++) {
        Map<String, Double> g = grads.get(zeroLabels.get(rIdx));
        for (int cIdx = 0; cIdx < quoteLabels.size(); cIdx++) {
          jac[rIdx][cIdx] = g.getOrDefault(quoteLabels.get(cIdx), 0.0);
        }
      }
      return new Result(curves, jac, zeroLabels, quoteLabels);
    }

    public List<String> quoteLabels() {
      return List.copyOf(quoteLabels);
    }

    @Override
    public void close() {
      mo.close();
    }
  }

  // ---- the recorded two-stage recursion --------------------------------

  private Map<String, SDouble> record(com.nablatensor.engine.AadRecorder rec) {
    Map<String, SDouble> out = new LinkedHashMap<>();

    // Stage 1 — OIS discount curve. Integer-year DFs indexed by year.
    Map<Integer, SDouble> oisDfByYear = new TreeMap<>();
    for (Inst inst : ois) {
      SDouble q = rec.input(inst.label(), inst.quote());
      double t = inst.maturity();
      SDouble df;
      if (inst.deposit()) {
        df = rec.constant(1.0).div(rec.constant(1.0).add(q.mul(t)));
      } else {
        SDouble prefix = rec.constant(0.0);
        for (var e : oisDfByYear.entrySet()) {
          if (e.getKey() <= t + 1e-9) {
            prefix = prefix.add(e.getValue());
          }
        }
        // annual par swap: P(0,T) = (1 - q * prefix_{<T}) / (1 + q)
        df = rec.constant(1.0).sub(q.mul(prefix)).div(rec.constant(1.0).add(q));
      }
      if (t == Math.rint(t)) {
        oisDfByYear.put((int) Math.rint(t), df);
      }
      out.put("z:" + inst.label(), df.log().neg().mul(1.0 / t));
    }

    // Stage 2 — one forecast curve per tenor, discounted on Stage 1.
    for (var entry : forecast.entrySet()) {
      Map<Integer, SDouble> fcDfByYear = new TreeMap<>();
      for (Inst inst : entry.getValue()) {
        SDouble q = rec.input(inst.label(), inst.quote());
        double t = inst.maturity();
        SDouble df;
        if (inst.deposit()) {
          df = rec.constant(1.0).div(rec.constant(1.0).add(q.mul(t)));
        } else {
          int n = (int) Math.rint(t);
          SDouble pdT = oisDfByYear.get(n);
          SDouble annuityT = rec.constant(0.0);
          for (int y = 1; y <= n; y++) {
            annuityT = annuityT.add(oisDfByYear.get(y));
          }
          // s_below = sum_{j=1}^{n-1} (P_fc(j-1)/P_fc(j) - 1) * P_d(j)
          SDouble sBelow = rec.constant(0.0);
          SDouble prevFc = rec.constant(1.0);
          for (int y = 1; y <= n - 1; y++) {
            SDouble pfc = fcDfByYear.get(y);
            sBelow = sBelow.add(prevFc.div(pfc).sub(1.0).mul(oisDfByYear.get(y)));
            prevFc = pfc;
          }
          SDouble prevFcLast = n == 1 ? rec.constant(1.0) : fcDfByYear.get(n - 1);
          // q*A_n = s_below + (P_fc(n-1)/x - 1) * P_d(n)
          //   =>  x = P_fc(n-1) * P_d(n) / (q*A_n - s_below + P_d(n))
          SDouble denom = q.mul(annuityT).sub(sBelow).add(pdT);
          df = prevFcLast.mul(pdT).div(denom);
          fcDfByYear.put(n, df);
        }
        if (t == Math.rint(t)) {
          fcDfByYear.putIfAbsent((int) Math.rint(t), df);
        }
        out.put("z:" + inst.label(), df.log().neg().mul(1.0 / t));
      }
    }
    return out;
  }

  private CurveSet assembleCurves(Map<String, Double> values) {
    YieldCurve discount = curveFrom(values, ois);
    Map<String, YieldCurve> fc = new LinkedHashMap<>();
    forecast.forEach((tenor, insts) -> fc.put(tenor, curveFrom(values, insts)));
    return new CurveSet(discount, fc);
  }

  private static YieldCurve curveFrom(Map<String, Double> values, List<Inst> insts) {
    double[] pillars = new double[insts.size()];
    double[] zeros = new double[insts.size()];
    for (int i = 0; i < insts.size(); i++) {
      pillars[i] = insts.get(i).maturity();
      zeros[i] = values.get("z:" + insts.get(i).label());
    }
    return new YieldCurve(pillars, zeros);
  }

  // ---- builder -------------------------------------------------------

  public static final class Builder {
    private final List<Inst> ois = new ArrayList<>();
    private final Map<String, List<Inst>> forecast = new LinkedHashMap<>();

    public Builder oisDeposit(double maturity, double rate) {
      return addOis(new Inst(OIS, true, maturity, rate));
    }

    public Builder oisSwap(double maturityYears, double rate) {
      requireInteger(maturityYears, "OIS swap");
      return addOis(new Inst(OIS, false, maturityYears, rate));
    }

    public Builder forecastDeposit(String tenor, double maturity, double rate) {
      return addForecast(tenor, new Inst(tenor, true, maturity, rate));
    }

    public Builder forecastSwap(String tenor, double maturityYears, double rate) {
      requireInteger(maturityYears, "forecast swap");
      return addForecast(tenor, new Inst(tenor, false, maturityYears, rate));
    }

    private Builder addOis(Inst inst) {
      if (!ois.isEmpty() && inst.maturity() <= ois.get(ois.size() - 1).maturity() + 1e-12) {
        throw new IllegalArgumentException("OIS instruments must be in strictly ascending maturity order");
      }
      ois.add(inst);
      return this;
    }

    private Builder addForecast(String tenor, Inst inst) {
      List<Inst> list = forecast.computeIfAbsent(tenor, k -> new ArrayList<>());
      if (!list.isEmpty() && inst.maturity() <= list.get(list.size() - 1).maturity() + 1e-12) {
        throw new IllegalArgumentException("forecast '" + tenor + "' instruments must ascend in maturity");
      }
      list.add(inst);
      return this;
    }

    public MultiCurveBootstrap build() {
      if (ois.isEmpty()) {
        throw new IllegalStateException("no OIS instruments");
      }
      int maxOisYear = 0;
      for (Inst i : ois) {
        if (i.maturity() == Math.rint(i.maturity())) {
          maxOisYear = Math.max(maxOisYear, (int) Math.rint(i.maturity()));
        }
      }
      // Every integer year up to the last OIS pillar must be present for the annual annuity.
      for (int y = 1; y <= maxOisYear; y++) {
        final int yy = y;
        if (ois.stream().noneMatch(i -> Math.abs(i.maturity() - yy) < 1e-9)) {
          throw new IllegalStateException("OIS curve is missing the integer-year pillar " + y
              + " needed for the annual annuity");
        }
      }
      for (var e : forecast.entrySet()) {
        int last = 0;
        for (Inst i : e.getValue()) {
          if (i.deposit()) {
            continue;
          }
          int n = (int) Math.rint(i.maturity());
          if (n > maxOisYear) {
            throw new IllegalStateException("forecast '" + e.getKey() + "' swap at " + n
                + "y exceeds the OIS curve (" + maxOisYear + "y)");
          }
          if (last != 0 && n != last + 1) {
            throw new IllegalStateException("forecast '" + e.getKey()
                + "' swaps must be consecutive integer years");
          }
          last = n;
        }
      }
      return new MultiCurveBootstrap(List.copyOf(ois),
          Map.copyOf(new LinkedHashMap<>(forecast)));
    }

    private static void requireInteger(double t, String what) {
      if (Math.abs(t - Math.rint(t)) > 1e-9) {
        throw new IllegalArgumentException(what + " must mature on an integer-year pillar; got " + t);
      }
    }
  }
}
