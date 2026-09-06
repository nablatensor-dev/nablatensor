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

import com.nablatensor.credit.CdoTranche;
import com.nablatensor.credit.CopulaMarket;
import com.nablatensor.credit.CopulaMonteCarlo;
import com.nablatensor.credit.PortfolioLossDistribution;
import com.nablatensor.engine.Nabla;
import java.util.Locale;

/**
 * Feature F9 — a synthetic CDO on a homogeneous pool. The tranche expected
 * losses come from the Andersen-Sidenius-Basu recursion; the recorded copula
 * Monte-Carlo reprices the equity tranche and returns its correlation delta and
 * default-probability sensitivity from one adjoint sweep.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.CdoTrancheShowcase}
 */
public final class CdoTrancheShowcase {

  private CdoTrancheShowcase() {
  }

  public static void main(String[] args) {
    double pd = 0.05;
    double lgd = 0.6;
    int names = 100;
    double rate = 0.02;
    double maturity = 1.0;

    double[][] tranches = {{0.0, 0.03}, {0.03, 0.07}, {0.07, 0.10}, {0.10, 0.15}, {0.15, 1.0}};

    System.out.printf(Locale.ROOT, "Synthetic CDO — %d names, PD %.0f%%, LGD %.0f%%, %.0fy%n%n",
        names, 100 * pd, 100 * lgd, maturity);
    System.out.printf(Locale.ROOT, "  tranche      EL fraction, by base correlation%n");
    System.out.printf(Locale.ROOT, "  %-12s %10s %10s %10s%n", "", "rho=0.15", "rho=0.30", "rho=0.45");
    for (double[] tr : tranches) {
      CdoTranche t = new CdoTranche(tr[0], tr[1]);
      System.out.printf(Locale.ROOT, "  %4.0f-%4.0f%%    %10.4f %10.4f %10.4f%n",
          100 * tr[0], 100 * tr[1],
          t.expectedLossFraction(PortfolioLossDistribution.homogeneous(pd, names, 0.15, lgd, 96)),
          t.expectedLossFraction(PortfolioLossDistribution.homogeneous(pd, names, 0.30, lgd, 96)),
          t.expectedLossFraction(PortfolioLossDistribution.homogeneous(pd, names, 0.45, lgd, 96)));
    }

    // Adjoint risk on the equity tranche from the recorded copula Monte-Carlo.
    CopulaMarket m = new CopulaMarket(0.30, pd);
    var equity = CopulaMonteCarlo.trancheLoss(0.0, 0.03, names, lgd, maturity, rate, 5e-3);
    try (Nabla.TypedPricer<CopulaMarket> p = Nabla.model(m, equity).fp64().greeks().on("cpu-jit").build()) {
      Nabla.TypedValuation<CopulaMarket> v = p.value().with(m).scenarios(1_000_000L).seed(42L).run();
      System.out.printf(Locale.ROOT,
          "%nEquity 0-3%% tranche, one adjoint sweep (rho=0.30, %,d paths):%n", v.scenarios());
      System.out.printf(Locale.ROOT, "  protection-leg PV   %+.6f  (per unit tranche notional)%n", v.price());
      System.out.printf(Locale.ROOT, "  d PV / d rho        %+.6f  (correlation delta, negative for equity)%n",
          v.greek(CopulaMarket::rho));
      System.out.printf(Locale.ROOT, "  d PV / d PD         %+.6f  (pool default-probability sensitivity)%n",
          v.greek(CopulaMarket::pd));
    }
  }
}
