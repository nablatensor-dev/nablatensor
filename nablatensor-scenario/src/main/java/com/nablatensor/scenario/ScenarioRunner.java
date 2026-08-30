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
package com.nablatensor.scenario;

import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.MultiOutput;
import com.nablatensor.engine.Nabla;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Expands a {@link ScenarioSet} onto {@code setInput} + replay of an
 * already-compiled kernel (Seam 6): declare the shocks as data, the runner moves
 * the market and re-prices without re-recording or recompiling.
 */
public final class ScenarioRunner {

  private ScenarioRunner() {
  }

  /** Runs every scenario against a built {@link MonteCarlo}, keyed by scenario name. */
  public static Map<String, Nabla.TypedValuation<EquityMarket>> run(MonteCarlo<EquityMarket> mc,
                                         EquityMarket base, ScenarioSet set, long scenarios, long seed) {
    Map<String, Nabla.TypedValuation<EquityMarket>> out = new LinkedHashMap<>();
    for (Scenario s : set.scenarios()) {
      out.put(s.name(), mc.run(shocked(base, s), scenarios, seed));
    }
    return out;
  }

  /** Runs every scenario against a built {@link MultiOutput}; each result carries all measures + Jacobians. */
  public static Map<String, MultiOutput.Result> run(MultiOutput mo, Map<String, Double> base,
                                                    ScenarioSet set, long scenarios, long seed) {
    Map<String, MultiOutput.Result> out = new LinkedHashMap<>();
    for (Scenario s : set.scenarios()) {
      out.put(s.name(), mo.run(diff(base, s.apply(base)), scenarios, seed));
    }
    return out;
  }

  /** A 1-D ladder result ready for plotting or a delta/gamma finite-difference. */
  public record LadderResult(String input, double[] x, double[] price, double[] delta) {}

  public static LadderResult ladder(MonteCarlo<EquityMarket> mc, EquityMarket base, Ladder ladder,
                                    long scenarios, long seed) {
    double[] x = ladder.values();
    double[] price = new double[x.length];
    double[] delta = new double[x.length];
    var scen = ScenarioSet.ladder(ladder).scenarios();
    for (int i = 0; i < x.length; i++) {
      Nabla.TypedValuation<EquityMarket> p = mc.run(shocked(base, scen.get(i)), scenarios, seed);
      price[i] = p.price();
      delta[i] = mc.hasGreeks() ? p.greek(EquityMarket::spot) : Double.NaN;
    }
    return new LadderResult(ladder.input(), x, price, delta);
  }

  /**
   * {@code base} with a scenario's shocks applied, staying inside the
   * {@link EquityMarket} type: each shock names a risk factor, and the matching
   * {@code with*} accessor is compiler-checked. A shock that names something
   * that is not an {@code EquityMarket} field is an error, not a silent no-op.
   */
  private static EquityMarket shocked(EquityMarket base, Scenario scenario) {
    EquityMarket m = base;
    for (Shock s : scenario.shocks()) {
      m = switch (s.input()) {
        case "spot"     -> m.withSpot(s.shocked(m.spot()));
        case "strike"   -> m.withStrike(s.shocked(m.strike()));
        case "vol"      -> m.withVol(s.shocked(m.vol()));
        case "rate"     -> m.withRate(s.shocked(m.rate()));
        case "maturity" -> m.withMaturity(s.shocked(m.maturity()));
        default -> throw new IllegalArgumentException("scenario '" + scenario.name()
            + "' shocks '" + s.input() + "', which is not an EquityMarket risk factor");
      };
    }
    return m;
  }

  private static Map<String, Double> diff(Map<String, Double> base, Map<String, Double> shocked) {
    Map<String, Double> d = new LinkedHashMap<>();
    shocked.forEach((k, v) -> {
      if (!v.equals(base.get(k))) {
        d.put(k, v);
      }
    });
    return d;
  }
}
