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
package com.nablatensor.examples;

import com.nablatensor.quant.BermudanLsm;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.OptionType;
import java.util.Locale;

/**
 * Feature F1 — least-squares Monte-Carlo for a Bermudan option by policy
 * optimisation: the per-date continuation value is a polynomial in log-moneyness
 * whose coefficients are chosen to maximise the price under the smoothed
 * exercise rule. The result is a lower bound, and its Greeks come from the same
 * reverse sweep (the envelope theorem makes the fitted coefficients "frozen"
 * for the market sensitivities).
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.BermudanLsmShowcase}
 */
public final class BermudanLsmShowcase {

  private BermudanLsmShowcase() {
  }

  public static void main(String[] args) {
    long paths = Long.getLong("paths", 150_000L);
    int dates = Integer.getInteger("dates", 25);

    // Longstaff-Schwartz (2001) Table 1 row: S=K=40, sigma=0.20, T=1, r=0.06;
    // the finite-difference American put value there is 2.314.
    EquityMarket m = new EquityMarket(40, 40, 0.20, 0.06, 1.0);

    long t0 = System.nanoTime();
    BermudanLsm.Result r = BermudanLsm.price(m, OptionType.PUT, dates, 6, 3, 0.6, paths, 42L);
    double s = (System.nanoTime() - t0) / 1e9;

    System.out.printf(Locale.ROOT, "American put by policy-optimisation LSM  (%d exercise dates, %,d paths, %.1fs)%n%n",
        dates, paths, s);
    System.out.printf(Locale.ROOT, "  European floor            %8.4f%n", r.europeanFloor());
    System.out.printf(Locale.ROOT, "  Bermudan (LSM lower bound) %8.4f   (+/- %.4f)%n", r.price(), r.standardError());
    System.out.printf(Locale.ROOT, "  early-exercise premium     %8.4f%n", r.earlyExercisePremium());
    System.out.printf(Locale.ROOT, "  reference (LS 2001, FD)    %8.4f%n%n", 2.314);
    System.out.printf(Locale.ROOT, "  from the same reverse sweep:  delta %+.4f   vega %+.4f   rho %+.4f%n",
        r.greeks().spot(), r.greeks().vol(), r.greeks().rate());
    System.out.printf(Locale.ROOT, "  optimiser: %d iterations, converged=%b%n", r.iterations(), r.converged());
  }
}
