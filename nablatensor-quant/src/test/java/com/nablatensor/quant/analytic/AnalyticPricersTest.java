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
package com.nablatensor.quant.analytic;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.quant.BlackScholes;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.OptionType;
import org.junit.jupiter.api.Test;

/**
 * Correctness of the analytic oracle layer (feature F2): cross-identities
 * between the members of the vanilla family, differenced Greeks pinned against
 * published Black-Scholes formulas, knock-in/out parity for the barriers, and
 * the credit-triangle limit for the CDS par spread.
 */
class AnalyticPricersTest {

  private static final double S = 100.0;
  private static final double K = 100.0;
  private static final double T = 1.0;
  private static final double R = 0.03;
  private static final double VOL = 0.20;

  @Test
  void generalizedBsmWithNoDividendMatchesBlackScholes() {
    EquityMarket m = new EquityMarket(S, K, VOL, R, T);
    // BlackScholes uses a lighter erfc approximation for N(x) (abs error ~1.5e-7),
    // so it is the coarser side of this comparison; tolerances reflect that.
    for (OptionType type : OptionType.values()) {
      BlackScholes ref = BlackScholes.of(type, m);
      GeneralizedBsm got = GeneralizedBsm.of(type, S, K, T, R, 0.0, VOL);

      assertEquals(ref.price(), got.price(), 5e-5, type + " price");
      assertEquals(ref.delta(), got.greeks().delta(), 5e-6, type + " delta");
      assertEquals(ref.vega(), got.greeks().vega(), 1e-4, type + " vega");
      assertEquals(ref.rho(), got.greeks().rho(), 1e-4, type + " rho");
      assertEquals(ref.strikeSensitivity(), got.greeks().strikeSensitivity(), 5e-6, type + " dV/dK");
    }
  }

  @Test
  void generalizedBsmGammaMatchesClosedForm() {
    double q = 0.01;
    double sqrtT = Math.sqrt(T);
    double b = R - q;
    double d1 = (Math.log(S / K) + (b + 0.5 * VOL * VOL) * T) / (VOL * sqrtT);
    double closedFormGamma = Math.exp(-q * T) * Normal.pdf(d1) / (S * VOL * sqrtT);

    AnalyticGreeks call = GeneralizedBsm.of(OptionType.CALL, S, K, T, R, q, VOL).greeks();
    AnalyticGreeks put = GeneralizedBsm.of(OptionType.PUT, S, K, T, R, q, VOL).greeks();

    assertEquals(closedFormGamma, call.gamma(), 1e-6, "call gamma");
    assertEquals(closedFormGamma, put.gamma(), 1e-6, "put gamma (same as call)");
  }

  @Test
  void black76EqualsBsmAtTheForward() {
    double q = 0.015;
    double forward = S * Math.exp((R - q) * T);
    for (OptionType type : OptionType.values()) {
      double black = Black76.price(type, forward, K, T, R, VOL);
      double bsm = GeneralizedBsm.of(type, S, K, T, R, q, VOL).price();
      assertEquals(bsm, black, 1e-10, type + ": Black-76 at the forward vs generalized BSM");
    }
  }

  @Test
  void garmanKohlhagenForeignRhoIsNegativeCarryRho() {
    double rd = 0.04;
    double rf = 0.012;
    GarmanKohlhagen call = GarmanKohlhagen.of(OptionType.CALL, 1.25, 1.20, 0.75, rd, rf, 0.11);
    // dV/dr_f by central difference on the price.
    double h = 1e-6;
    double up = GarmanKohlhagen.of(OptionType.CALL, 1.25, 1.20, 0.75, rd, rf + h, 0.11).price();
    double dn = GarmanKohlhagen.of(OptionType.CALL, 1.25, 1.20, 0.75, rd, rf - h, 0.11).price();
    assertEquals((up - dn) / (2 * h), call.foreignRho(), 1e-4, "foreign rho");
    assertTrue(call.price() > 0.0, "positive price");
  }

  @Test
  void bachelierParityAndKnownAtmValue() {
    double f = 100.0;
    double normalVol = 10.0;
    double call = Bachelier.price(OptionType.CALL, f, f, T, 0.0, normalVol);
    double put = Bachelier.price(OptionType.PUT, f, f, T, 0.0, normalVol);
    // ATM: call = put = sigmaN sqrt(T) / sqrt(2 pi)
    double atm = normalVol * Math.sqrt(T) / Math.sqrt(2.0 * Math.PI);
    assertEquals(atm, call, 1e-10, "Bachelier ATM call");
    assertEquals(call, put, 1e-12, "Bachelier ATM call = put");

    // Put-call parity with a struck option and discounting.
    double kk = 105.0;
    double disc = Math.exp(-R * T);
    double c = Bachelier.price(OptionType.CALL, f, kk, T, R, normalVol);
    double p = Bachelier.price(OptionType.PUT, f, kk, T, R, normalVol);
    assertEquals(disc * (f - kk), c - p, 1e-10, "Bachelier put-call parity");
  }

  @Test
  void mertonCollapsesToBsmWhenIntensityIsZero() {
    for (OptionType type : OptionType.values()) {
      double merton = MertonJumpDiffusion.price(type, S, K, T, R, VOL, 0.0, -0.1, 0.15);
      double bsm = GeneralizedBsm.of(type, S, K, T, R, 0.0, VOL).price();
      assertEquals(bsm, merton, 1e-10, type + ": Merton with lambda=0 vs BSM");
    }
  }

  @Test
  void mertonAddsValueAndKeepsParity() {
    double lambda = 0.75;
    double muJ = -0.05;
    double deltaJ = 0.18;
    double call = MertonJumpDiffusion.price(OptionType.CALL, S, K, T, R, VOL, lambda, muJ, deltaJ);
    double put = MertonJumpDiffusion.price(OptionType.PUT, S, K, T, R, VOL, lambda, muJ, deltaJ);
    double bsmCall = GeneralizedBsm.of(OptionType.CALL, S, K, T, R, 0.0, VOL).price();

    assertTrue(call > bsmCall, "jumps add value to an ATM call");
    // Risk-neutral drift is preserved, so the forward is still S e^{rT}.
    assertEquals(S - K * Math.exp(-R * T), call - put, 1e-8, "Merton put-call parity");
  }

  @Test
  void barrierKnockInOutParityForAllEightCombinations() {
    double h = 0.20;
    for (OptionType type : OptionType.values()) {
      double vanilla = com.nablatensor.quant.analytic.CostOfCarry.price(type, S, K, T, R, R, VOL);

      double di = BarrierAnalytic.price(type, BarrierAnalytic.Kind.DOWN_IN, S, K, 90.0, T, R, R, VOL);
      double doo = BarrierAnalytic.price(type, BarrierAnalytic.Kind.DOWN_OUT, S, K, 90.0, T, R, R, VOL);
      double ui = BarrierAnalytic.price(type, BarrierAnalytic.Kind.UP_IN, S, K, 115.0, T, R, R, VOL);
      double uo = BarrierAnalytic.price(type, BarrierAnalytic.Kind.UP_OUT, S, K, 115.0, T, R, R, VOL);

      assertEquals(vanilla, di + doo, 1e-9, type + ": down in + down out = vanilla");
      assertEquals(vanilla, ui + uo, 1e-9, type + ": up in + up out = vanilla");
      assertTrue(di >= -1e-12 && doo >= -1e-12 && ui >= -1e-12 && uo >= -1e-12,
          type + ": barrier prices non-negative");
      // silence unused-variable inspection while keeping the intent explicit
      assertTrue(h > 0.0);
    }
  }

  @Test
  void barrierUnreachableRecoversTheVanilla() {
    double vanillaCall = CostOfCarry.price(OptionType.CALL, S, K, T, R, R, VOL);
    // An up-and-out call with the barrier far above any plausible path ~ vanilla.
    double farUp = BarrierAnalytic.price(OptionType.CALL, BarrierAnalytic.Kind.UP_OUT,
        S, K, 100_000.0, T, R, R, VOL);
    assertEquals(vanillaCall, farUp, 1e-6, "unreachable up-and-out call = vanilla call");

    // A down-and-out put with the barrier far below ~ vanilla put.
    double vanillaPut = CostOfCarry.price(OptionType.PUT, S, K, T, R, R, VOL);
    double farDown = BarrierAnalytic.price(OptionType.PUT, BarrierAnalytic.Kind.DOWN_OUT,
        S, K, 1e-6, T, R, R, VOL);
    assertEquals(vanillaPut, farDown, 1e-6, "unreachable down-and-out put = vanilla put");
  }

  @Test
  void barrierDownOutCallFallsAsBarrierRises() {
    double last = Double.POSITIVE_INFINITY;
    for (double h = 80.0; h <= 99.0; h += 1.0) {
      double v = BarrierAnalytic.price(OptionType.CALL, BarrierAnalytic.Kind.DOWN_OUT,
          S, K, h, T, R, R, VOL);
      assertTrue(v <= last + 1e-12, "down-and-out call value decreases as the barrier approaches spot");
      last = v;
    }
  }

  @Test
  void cdsParSpreadMatchesCreditTriangle() {
    double lambda = 0.02;
    double recovery = 0.4;
    double rate = 0.025;
    int months = 60;
    double[] times = new double[months];
    double[] survival = new double[months];
    double[] discount = new double[months];
    for (int i = 0; i < months; i++) {
      double t = (i + 1) / 12.0;
      times[i] = t;
      survival[i] = Math.exp(-lambda * t);
      discount[i] = Math.exp(-rate * t);
    }

    CdsParSpread cds = CdsParSpread.of(recovery, times, survival, discount);
    double creditTriangle = lambda * (1.0 - recovery);
    assertEquals(creditTriangle, cds.parSpread(), 0.05 * creditTriangle, "par spread ~ lambda (1 - R)");
    assertEquals(0.0, cds.protectionBuyerValue(cds.parSpread()), 1e-12, "par contract has zero value");
  }

  @Test
  void inverseCdfRoundTrips() {
    assertEquals(0.0, Normal.inverseCdf(0.5), 1e-12);
    for (double p = 0.001; p < 1.0; p += 0.037) {
      double x = Normal.inverseCdf(p);
      assertEquals(p, Normal.cdf(x), 1e-12, "cdf(inverseCdf(p)) = p at p=" + p);
      assertEquals(-x, Normal.inverseCdf(1.0 - p), 1e-9, "symmetry at p=" + p);
    }
  }
}
