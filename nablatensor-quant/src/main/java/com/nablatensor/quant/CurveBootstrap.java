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

import java.util.ArrayList;
import java.util.List;

/**
 * Sequential single-curve bootstrap from cash deposits, FRAs and annual par
 * swaps, plus the <em>analytic</em> Jacobian {@code d(zeroRate_i)/d(quote_j)}.
 *
 * <p>Each instrument introduces exactly one new pillar (its maturity), solved in
 * closed form:
 * <ul>
 *   <li>deposit: {@code P(0,T) = 1 / (1 + r T)}</li>
 *   <li>FRA {@code [t1,t2]}: {@code P(0,t2) = P(0,t1) / (1 + r (t2 - t1))}</li>
 *   <li>par swap: {@code P(0,T) = (1 - r * annuity_{<T}) / (1 + r * tau)}</li>
 * </ul>
 * and the Jacobian is the exact derivative of that recursion, chained through
 * the earlier discount factors (lower triangular). Deposits and FRAs sit at the
 * short end; par swaps must land on consecutive integer-year pillars so every
 * fixed-leg payment coincides with a pillar (no interpolated payment DFs).
 * Multi-curve (OIS discounting + tenor basis) is Phase 2.
 */
public final class CurveBootstrap {

  private sealed interface Instrument permits Deposit, Fra, Swap {
    double maturity();
    double quote();
  }

  private record Deposit(double maturity, double quote) implements Instrument {}

  private record Fra(double start, double maturity, double quote) implements Instrument {}

  private record Swap(double maturity, double quote) implements Instrument {}

  private final List<Instrument> instruments;
  private double[] pillars;
  private double[] zeroRates;
  private double[] df;
  private double[][] dZeroDQuote;

  private CurveBootstrap(List<Instrument> instruments) {
    this.instruments = List.copyOf(instruments);
    solve();
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Convenience: annual par swaps for maturities {@code 1..n} years. */
  public static CurveBootstrap fromAnnualParSwaps(double[] parRates) {
    Builder b = builder();
    for (int i = 0; i < parRates.length; i++) {
      b.swap(i + 1.0, parRates[i]);
    }
    return b.build();
  }

  public YieldCurve curve() {
    return new YieldCurve(pillars.clone(), zeroRates.clone());
  }

  /** {@code J[i][j] = d zeroRate_i / d quote_j}; lower triangular in instrument order. */
  public double[][] zeroRateJacobian() {
    double[][] copy = new double[dZeroDQuote.length][];
    for (int i = 0; i < copy.length; i++) {
      copy[i] = dZeroDQuote[i].clone();
    }
    return copy;
  }

  public double[] pillars() {
    return pillars.clone();
  }

  private void solve() {
    int n = instruments.size();
    pillars = new double[n];
    zeroRates = new double[n];
    df = new double[n];
    dZeroDQuote = new double[n][n];

    for (int i = 0; i < n; i++) {
      Instrument inst = instruments.get(i);
      pillars[i] = inst.maturity();
      double[] dP = new double[n];      // d(df[i]) / d(quote_j)
      double p;

      switch (inst) {
        case Deposit d -> {
          double den = 1.0 + d.quote() * d.maturity();
          p = 1.0 / den;
          dP[i] = -d.maturity() / (den * den);
        }
        case Fra f -> {
          int startIdx = pillarIndex(f.start());
          double pStart = startIdx < 0 ? 1.0 : df[startIdx];   // FRA can start at 0
          double delta = f.maturity() - f.start();
          double den = 1.0 + f.quote() * delta;
          p = pStart / den;
          if (startIdx >= 0) {
            for (int j = 0; j <= startIdx; j++) {
              dP[j] += (-pillars[startIdx] * df[startIdx] * dZeroDQuote[startIdx][j]) / den;
            }
          }
          dP[i] += -pStart * delta / (den * den);
        }
        case Swap s -> {
          double prefix = 0.0;
          double[] dPrefix = new double[n];
          for (int k = 0; k < i; k++) {
            if (pillars[k] <= s.maturity() + 1e-9 && isYearGrid(pillars[k])) {
              prefix += df[k];
              for (int j = 0; j <= k; j++) {
                dPrefix[j] += -pillars[k] * df[k] * dZeroDQuote[k][j];
              }
            }
          }
          double tau = 1.0;   // annual accrual for the newly introduced payment
          double den = 1.0 + s.quote() * tau;
          double num = 1.0 - s.quote() * prefix;
          p = num / den;
          for (int j = 0; j < i; j++) {
            dP[j] = (-s.quote() * dPrefix[j]) / den;
          }
          dP[i] = (-prefix * den - num * tau) / (den * den);
        }
        default -> throw new IllegalStateException();
      }

      df[i] = p;
      zeroRates[i] = -Math.log(p) / pillars[i];
      double factor = -1.0 / (p * pillars[i]);
      for (int j = 0; j <= i; j++) {
        dZeroDQuote[i][j] = factor * dP[j];
      }
    }
  }

  private int pillarIndex(double t) {
    for (int i = 0; i < pillars.length; i++) {
      if (Math.abs(pillars[i] - t) < 1e-9) {
        return i;
      }
    }
    return -1;
  }

  private static boolean isYearGrid(double t) {
    return Math.abs(t - Math.rint(t)) < 1e-9;
  }

  /** Declares the calibrating instruments in ascending maturity order. */
  public static final class Builder {
    private final List<Instrument> instruments = new ArrayList<>();

    public Builder deposit(double maturity, double rate) {
      return add(new Deposit(maturity, rate));
    }

    public Builder fra(double start, double end, double rate) {
      return add(new Fra(start, end, rate));
    }

    public Builder swap(double maturity, double rate) {
      if (Math.abs(maturity - Math.rint(maturity)) > 1e-9) {
        throw new IllegalArgumentException("par swaps must mature on an integer-year pillar; got " + maturity);
      }
      boolean prevIsSwap = !instruments.isEmpty()
          && instruments.get(instruments.size() - 1) instanceof Swap;
      if (prevIsSwap
          && Math.abs(maturity - instruments.get(instruments.size() - 1).maturity() - 1.0) > 1e-9) {
        throw new IllegalArgumentException("consecutive par swaps must be one year apart");
      }
      return add(new Swap(maturity, rate));
    }

    private Builder add(Instrument inst) {
      if (!instruments.isEmpty()
          && inst.maturity() <= instruments.get(instruments.size() - 1).maturity() + 1e-12) {
        throw new IllegalArgumentException("instruments must be in strictly ascending maturity order");
      }
      instruments.add(inst);
      return this;
    }

    public CurveBootstrap build() {
      if (instruments.isEmpty()) {
        throw new IllegalStateException("no instruments");
      }
      return new CurveBootstrap(instruments);
    }
  }
}
