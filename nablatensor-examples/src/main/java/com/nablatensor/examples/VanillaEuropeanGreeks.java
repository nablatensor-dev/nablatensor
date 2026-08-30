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

import com.nablatensor.quant.BlackScholes;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.OptionType;
import com.nablatensor.quant.Products;
import com.nablatensor.engine.Nabla;
import java.util.Locale;

/**
 * Vanilla European call: price and every first-order Greek from one adjoint
 * sweep, checked against the Black-Scholes closed form.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.VanillaEuropeanGreeks}
 */
public final class VanillaEuropeanGreeks {

  private VanillaEuropeanGreeks() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    long scenarios = Long.getLong("scenarios", 2_000_000L);
    long seed = Long.getLong("seed", 42L);

    try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.europeanCall())
        .market(market)
        .steps(1)                 // the terminal value is all a European needs
        .fp64()
        .greeks()
        .on("cpu-jit")            // the LTS-clean default: no native lib, no incubator flag
        .build()) {

      Nabla.TypedValuation<EquityMarket> p = mc.run(scenarios, seed);
      BlackScholes bs = BlackScholes.of(OptionType.CALL, market);

      System.out.printf(Locale.ROOT, "engine=%s  tape=%d nodes  record=%.1f ms  build=%.1f ms%n",
          mc.engine(), mc.nodes(), mc.recordSeconds() * 1e3, mc.buildSeconds() * 1e3);
      System.out.printf(Locale.ROOT, "%,d scenarios in %.3f s  (%.2e scenarios/s)%n%n",
          p.scenarios(), p.seconds(), p.scenariosPerSecond());

      System.out.printf(Locale.ROOT, "%-8s %16s %16s %14s%n", "", "adjoint MC", "Black-Scholes", "abs error");
      line("price", p.price(), bs.price());
      line("delta", p.greek(EquityMarket::spot), bs.delta());
      line("vega", p.greek(EquityMarket::vol), bs.vega());
      line("rho", p.greek(EquityMarket::rate), bs.rho());
      line("dV/dK", p.greek(EquityMarket::strike), bs.strikeSensitivity());
    }
  }

  private static void line(String name, double mc, double ref) {
    System.out.printf(Locale.ROOT, "%-8s %16.6f %16.6f %14.2e%n", name, mc, ref, Math.abs(mc - ref));
  }
}
