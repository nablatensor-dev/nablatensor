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

import com.nablatensor.engine.AadEngine;
import com.nablatensor.engine.AadEngines;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.examples.FrtbFullShowcase.Capital;
import com.nablatensor.examples.FrtbFullShowcase.CurvatureRun;
import com.nablatensor.examples.FrtbFullShowcase.DrcResult;
import com.nablatensor.examples.FrtbFullShowcase.MarketData;
import com.nablatensor.examples.FrtbFullShowcase.ParameterSet;
import com.nablatensor.examples.FrtbFullShowcase.RraoResult;
import com.nablatensor.examples.FrtbFullShowcase.SignOff;
import com.nablatensor.examples.FrtbFullShowcase.TradeSpec;
import com.nablatensor.risk.Portfolio;
import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.Sensitivities;
import java.util.List;
import java.util.Locale;

/**
 * The whole FRTB standardised approach — the same calculation the
 * {@code frtb-full-on-cuda} demo and notebook walk through — run end to end on
 * whichever backend this machine can drive fastest.
 *
 * <p>Where {@link FrtbFullShowcase} narrates all eight stages on an engine you
 * name, this one picks the engine itself: the highest-priority engine the
 * {@code ServiceLoader} found that is actually usable here, preferring FP32
 * because that is the throughput case on every accelerator. On a CUDA box that
 * resolves to CUDA; on a laptop with no GPU it degrades to {@code cpu-jit} and
 * still produces a capital number, only slower.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.FrtbFullOnFastestBackend}
 *
 * <p>System properties: {@code -Dengine=} pins one backend instead of choosing,
 * {@code -Dscenarios=}, {@code -Dsteps=}, {@code -Dseed=}.
 */
public final class FrtbFullOnFastestBackend {

  /** One adjoint sweep plus base/up/down repricings — what a single CVR costs. */
  private static final int PASSES_PER_CURVATURE = 4;

  private FrtbFullOnFastestBackend() {
  }

  public static void main(String[] args) {
    long scenarios = Long.getLong("scenarios", 250_000L);
    int steps = Integer.getInteger("steps", 252);
    long seed = Long.getLong("seed", 42L);

    Choice choice = fastestUsable(System.getProperty("engine"));
    printBanner(choice, scenarios, steps, seed);

    ParameterSet parameters = FrtbFullShowcase.parameters();
    MarketData marketData = FrtbFullShowcase.marketData();
    List<TradeSpec> tradeSpecs = FrtbFullShowcase.trades();

    CurvatureRun heavy = FrtbFullShowcase.runHeavyCurvature(
        choice.engine().name(), choice.fp32(), scenarios, steps, seed);
    Portfolio portfolio =
        FrtbFullShowcase.buildPortfolio(tradeSpecs, parameters, marketData, heavy.cvr());
    Sensitivities netted = portfolio.aggregate();
    Capital capital = FrtbFullShowcase.aggregate(parameters, netted);
    DrcResult drc = FrtbFullShowcase.defaultRiskCharge(FrtbFullShowcase.defaultPositions());
    RraoResult rrao =
        FrtbFullShowcase.residualRiskAddOn(FrtbFullShowcase.residualPositions(tradeSpecs));
    SignOff signOff =
        SignOff.review(parameters, marketData, portfolio, capital, drc, rrao);

    printCurvature(heavy);
    printCapital(netted, capital, drc, rrao, signOff);
    printThroughput(heavy, scenarios);
  }

  /**
   * Highest-priority engine that is usable here, FP32 first. {@code discovered()}
   * is already sorted by descending priority, and {@code available(...)} is the
   * safe probe — an engine whose runtime is missing must not throw out of here.
   */
  static Choice fastestUsable(String requested) {
    List<String> fp32 = usableNames(AadOptions.Precision.FLOAT32);
    List<String> fp64 = usableNames(AadOptions.Precision.FLOAT64);
    for (AadEngine engine : AadEngines.discovered()) {
      if (requested != null && !requested.isBlank() && !engine.name().equalsIgnoreCase(requested)) {
        continue;
      }
      if (fp32.contains(engine.name())) {
        return new Choice(engine, true);
      }
      if (fp64.contains(engine.name())) {
        return new Choice(engine, false);
      }
    }
    throw new IllegalStateException("no usable AAD engine here"
        + (requested == null ? "" : " for -Dengine=" + requested)
        + "; fp32: " + fp32 + ", fp64: " + fp64);
  }

  private static List<String> usableNames(AadOptions.Precision precision) {
    return AadEngines.available(new AadOptions(precision, true)).stream()
        .map(AadEngine::name)
        .toList();
  }

  private static void printBanner(Choice choice, long scenarios, int steps, long seed) {
    System.out.printf(Locale.ROOT, "FRTB FULL on the fastest backend available here%n");
    System.out.printf(Locale.ROOT, "  engine     %s (priority %d, %s)%n",
        choice.engine().name(), choice.engine().priority(), choice.fp32() ? "fp32" : "fp64");
    System.out.printf(Locale.ROOT, "  runs on    %s%n", choice.engine().describe());
    System.out.printf(Locale.ROOT, "  workload   %,d paths x %d fixings, seed %d%n",
        scenarios, steps, seed);
    System.out.printf(Locale.ROOT, "  also here  %s%n%n", others(choice));
  }

  private static String others(Choice choice) {
    List<String> rest = AadEngines.discovered().stream()
        .map(AadEngine::name)
        .filter(name -> !name.equals(choice.engine().name()))
        .toList();
    return rest.isEmpty() ? "(none)" : String.join(", ", rest);
  }

  private static void printCurvature(CurvatureRun heavy) {
    System.out.println("HEAVY STAGE  adjoint delta + three full shocked repricings");
    System.out.printf(Locale.ROOT, "  base PV    %12.6f%n", heavy.base());
    System.out.printf(Locale.ROOT, "  up / down  %12.6f / %.6f%n", heavy.up(), heavy.down());
    System.out.printf(Locale.ROOT, "  delta      %12.6f   shock %.6f%n",
        heavy.delta(), heavy.shock());
    System.out.printf(Locale.ROOT, "  CVR        %12.6f%n%n", heavy.cvr());
  }

  private static void printCapital(Sensitivities netted, Capital capital,
                                   DrcResult drc, RraoResult rrao, SignOff signOff) {
    System.out.printf(Locale.ROOT, "NETTED BOOK  %d shared risk factors across %d classes%n",
        netted.asMap().size(), RiskClass.values().length);
    System.out.printf(Locale.ROOT, "  SBM        %10.4f $m%n", capital.total());
    System.out.printf(Locale.ROOT, "  DRC        %10.4f $m%n", drc.total());
    System.out.printf(Locale.ROOT, "  RRAO       %10.4f $m%n", rrao.total());
    System.out.printf(Locale.ROOT, "  TOTAL      %10.4f $m   sign-off: %s%n%n",
        capital.total() + drc.total() + rrao.total(), signOff.status());
  }

  private static void printThroughput(CurvatureRun heavy, long scenarios) {
    double seconds = heavy.seconds();
    long paths = PASSES_PER_CURVATURE * scenarios;
    System.out.printf(Locale.ROOT,
        "ENGINE WORK  %.3f s for %,d path valuations (%.2e paths/s)%n",
        seconds, paths, seconds > 0.0 ? paths / seconds : Double.NaN);
    System.out.printf(Locale.ROOT,
        "  adjoint %.3f s   base %.3f s   up %.3f s   down %.3f s%n",
        heavy.adjointSeconds(), heavy.baseSeconds(), heavy.upSeconds(), heavy.downSeconds());
    System.out.println("Everything after the repricings is scalar post-processing.");
  }

  record Choice(AadEngine engine, boolean fp32) {
  }
}
