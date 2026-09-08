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
package com.nablatensor.validate;

import com.nablatensor.quant.EquityMarket;
import com.nablatensor.engine.Nabla;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * The result of a {@link ModelValidation} run, rendered as a plain-text evidence
 * pack suitable for a model-validation file (SR 11-7 / TRIM / SS1/23 style):
 * reproducible inputs, an independent reference, and a pass/fail line per
 * backend.
 */
public record Report(String product, EquityMarket market, int steps, long scenarios, long seed,
                     boolean fp32, double tolerance, String machine, Nabla.TypedValuation<EquityMarket> oracle,
                     List<EngineComparison> comparisons, BumpCrossCheck bumpCrossCheck,
                     AnalyticComparison analytic) {

  public boolean passed() {
    return comparisons.stream().allMatch(EngineComparison::withinTolerance);
  }

  public String firstFailure() {
    return comparisons.stream()
        .filter(c -> !c.withinTolerance())
        .map(c -> c.engine() + ": priceRel=" + c.priceRelDiff() + " gradRel=" + c.gradMaxRelDiff())
        .findFirst()
        .orElse("(none)");
  }

  public void writeText(Path file) throws IOException {
    Files.writeString(file, toString());
  }

  @Override
  public String toString() {
    StringBuilder b = new StringBuilder();
    Locale l = Locale.ROOT;
    b.append("NablaTensor — model-validation evidence pack\n");
    b.append("============================================\n\n");
    b.append(String.format(l, "product        : %s%n", product));
    b.append(String.format(l, "market         : S0=%.4f K=%.4f sigma=%.4f r=%.4f T=%.4f%n",
        market.spot(), market.strike(), market.vol(), market.rate(), market.maturity()));
    b.append(String.format(l, "discretisation : %d steps%n", steps));
    b.append(String.format(l, "scenarios      : %,d%n", scenarios));
    b.append(String.format(l, "seed           : 0x%016X%n", seed));
    b.append(String.format(l, "precision      : %s%n", fp32 ? "fp32" : "fp64"));
    b.append(String.format(l, "tolerance      : %.2e (relative)%n", tolerance));
    b.append(String.format(l, "machine        : %s%n", machine));
    b.append('\n');

    b.append("scalar CPU oracle (reference)\n");
    b.append("-----------------------------\n");
    b.append(String.format(l, "  price   %+.10e   (%,d scenarios in %.3f s)%n",
        oracle.price(), oracle.scenarios(), oracle.seconds()));
    {
      EquityMarket g = oracle.greeks();
      b.append(String.format(l, "  delta   %+.10e%n", g.spot()));
      b.append(String.format(l, "  vega    %+.10e%n", g.vol()));
      b.append(String.format(l, "  rho     %+.10e%n", g.rate()));
      b.append(String.format(l, "  dV/dK   %+.10e%n", g.strike()));
      b.append(String.format(l, "  dV/dT   %+.10e%n", g.maturity()));
    }
    b.append('\n');

    b.append("backend reproduction vs oracle (equal seed, equal scenarios)\n");
    b.append("-----------------------------------------------------------\n");
    b.append(String.format(l, "  %-10s  %-6s  %14s  %14s  %s%n",
        "engine", "result", "price relΔ", "grad relΔ", "detail"));
    for (EngineComparison c : comparisons) {
      b.append(String.format(l, "  %-10s  %-6s  %14.3e  %14.3e  %s%n",
          c.engine(), c.withinTolerance() ? "PASS" : "FAIL",
          c.priceRelDiff(), c.gradMaxRelDiff(), c.describe()));
    }
    if (comparisons.isEmpty()) {
      b.append("  (no accelerated backend available on this machine; oracle only)\n");
    }
    b.append('\n');

    b.append("adjoint gradient vs central bump-and-revalue on the oracle\n");
    b.append("---------------------------------------------------------\n");
    BumpCrossCheck x = bumpCrossCheck;
    b.append(String.format(l, "  bump size      : %.2e (relative, common random numbers)%n", x.relativeBump()));
    b.append(String.format(l, "  %-8s  %16s  %16s  %12s%n", "greek", "adjoint", "bump", "absΔ"));
    row(b, l, "delta", x.adjoint().spot(), x.bump().spot(), x.absDiff().spot());
    row(b, l, "dV/dK", x.adjoint().strike(), x.bump().strike(), x.absDiff().strike());
    row(b, l, "vega", x.adjoint().vol(), x.bump().vol(), x.absDiff().vol());
    row(b, l, "rho", x.adjoint().rate(), x.bump().rate(), x.absDiff().rate());
    row(b, l, "dV/dT", x.adjoint().maturity(), x.bump().maturity(), x.absDiff().maturity());
    b.append('\n');

    if (analytic != null) {
      b.append("adjoint price and gradient vs an independent closed form\n");
      b.append("------------------------------------------------------\n");
      EquityMarket g = oracle.greeks();
      b.append(String.format(l, "  %-8s  %16s  %16s  %12s%n", "quantity", "adjoint", "closed form", "absΔ"));
      arow(b, l, "price", oracle.price(), analytic.reference().price(), analytic.priceAbsDiff());
      arow(b, l, "delta", g.spot(), analytic.reference().delta(), analytic.deltaAbsDiff());
      arow(b, l, "vega", g.vol(), analytic.reference().vega(), analytic.vegaAbsDiff());
      arow(b, l, "rho", g.rate(), analytic.reference().rho(), analytic.rhoAbsDiff());
      arow(b, l, "dV/dK", g.strike(), analytic.reference().strikeSensitivity(), analytic.strikeAbsDiff());
      b.append(String.format(l, "  band: 3·SE + 5e-3·|ref|  =>  %s%n", analytic.withinBand() ? "PASS" : "FAIL"));
      b.append('\n');
    }

    b.append(passed() ? "RESULT: PASS — every backend reproduces the oracle within tolerance.\n"
                      : "RESULT: FAIL — " + firstFailure() + "\n");
    return b.toString();
  }

  private static void arow(StringBuilder b, Locale l, String name, double adj, double ref, double diff) {
    b.append(String.format(l, "  %-8s  %+16.8e  %+16.8e  %12.2e%n", name, adj, ref, diff));
  }

  private static void row(StringBuilder b, Locale l, String name, double adj, double bump, double diff) {
    b.append(String.format(l, "  %-8s  %+16.8e  %+16.8e  %12.2e%n", name, adj, bump, diff));
  }
}
