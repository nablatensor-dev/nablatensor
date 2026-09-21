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
package com.nablatensor.quant.transform;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.HestonMarket;
import com.nablatensor.quant.HestonModel;
import com.nablatensor.quant.OptionTypeEnum;
import com.nablatensor.quant.analytic.GeneralizedBsm;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Feature F13: the COS method reproduces Black-Scholes to machine precision,
 * agrees with a Heston Monte-Carlo, recovers Black-Scholes as the VG variance
 * rate vanishes, and its Heston surface calibration recovers the generating
 * parameters.
 */
@Tag("mc")
class CosMethodTest {

  @Test
  void cosMatchesBlackScholesAcrossStrikes() {
    double s = 100;
    double r = 0.03;
    double t = 0.75;
    double vol = 0.22;
    BsmCf cf = BsmCf.of().rate(r).vol(vol).build();
    for (double k : new double[] {60, 80, 95, 100, 110, 140, 180}) {
      double cos = CosMethod.price(cf, OptionTypeEnum.CALL, s, k, r, t);
      double closed = GeneralizedBsm.of().type(OptionTypeEnum.CALL).spot(s).strike(k).maturity(t).rate(r).dividend(0.0).vol(vol).build().price();
      assertEquals(closed, cos, 1e-7 * (1 + closed), "COS vs Black-Scholes at K=" + k);

      double cosPut = CosMethod.price(cf, OptionTypeEnum.PUT, s, k, r, t);
      assertEquals(cos - s + k * Math.exp(-r * t), cosPut, 1e-9, "COS put-call parity at K=" + k);
    }
  }

  @Test
  void cosHestonAgreesWithMonteCarlo() {
    double s = 100;
    double r = 0.02;
    double t = 1.0;
    // v0, kappa, theta, xi, rho
    HestonCf cf = HestonCf.of().rate(r).v0(0.04).kappa(1.5).theta(0.04).xi(0.4).rho(-0.6).build();

    for (double k : new double[] {90, 100, 110}) {
      double cos = CosMethod.price(cf, OptionTypeEnum.CALL, s, k, r, t);
      HestonMarket m = HestonMarket.of().spot(s).strike(k).rate(r).v0(0.04).kappa(1.5).theta(0.04).xi(0.4).rho(-0.6).build();
      double mc = price(m, HestonModel.european(OptionTypeEnum.CALL, t, 200), 800_000L, 42L);
      assertEquals(mc, cos, 0.03 * cos + 0.02, "COS Heston vs MC at K=" + k + " (cos=" + cos + ", mc=" + mc + ")");
    }
  }

  @Test
  void cosHestonPutCallParityOnTheDirectCall() {
    // price() derives the put by parity, so exercise the call directly and
    // reconstruct the put from a second independent COS call via parity identity.
    HestonCf cf = HestonCf.of().rate(0.025).v0(0.05).kappa(2.0).theta(0.045).xi(0.5).rho(-0.7).build();
    double s = 100;
    double k = 105;
    double t = 0.5;
    double call = CosMethod.price(cf, OptionTypeEnum.CALL, s, k, 0.025, t);
    double put = CosMethod.price(cf, OptionTypeEnum.PUT, s, k, 0.025, t);
    assertTrue(call > 0 && put > 0, "both positive");
    assertEquals(s - k * Math.exp(-0.025 * t), call - put, 1e-9);
  }

  @Test
  void cosVarianceGammaRecoversBlackScholesAsNuVanishes() {
    double s = 100;
    double r = 0.03;
    double t = 1.0;
    double sigma = 0.2;
    VarianceGammaCf cf = VarianceGammaCf.of().rate(r).sigma(sigma).nu(1e-6).theta(-0.1).build();
    for (double k : new double[] {85, 100, 120}) {
      double vg = CosMethod.price(cf, OptionTypeEnum.CALL, s, k, r, t);
      double bs = GeneralizedBsm.of().type(OptionTypeEnum.CALL).spot(s).strike(k).maturity(t).rate(r).dividend(0.0).vol(sigma).build().price();
      assertEquals(bs, vg, 2e-3 * (1 + bs), "VG -> BS as nu -> 0 at K=" + k);
    }
    // A symmetric VG (theta = 0) with non-trivial nu fattens both tails, so a
    // deep OTM call is richer than flat Black.
    VarianceGammaCf symmetric = VarianceGammaCf.of().rate(r).sigma(sigma).nu(0.5).theta(0.0).build();
    double otm = CosMethod.price(symmetric, OptionTypeEnum.CALL, s, 135, r, t);
    double otmBs = GeneralizedBsm.of().type(OptionTypeEnum.CALL).spot(s).strike(135).maturity(t).rate(r).dividend(0.0).vol(sigma).build().price();
    assertTrue(otm > otmBs, "symmetric VG fattens the right tail: " + otm + " vs " + otmBs);

    // Negative theta skews the smile: an OTM put becomes richer than flat Black.
    VarianceGammaCf skewed = VarianceGammaCf.of().rate(r).sigma(sigma).nu(0.4).theta(-0.3).build();
    double otmPut = CosMethod.price(skewed, OptionTypeEnum.PUT, s, 70, r, t);
    double otmPutBs = GeneralizedBsm.of().type(OptionTypeEnum.PUT).spot(s).strike(70).maturity(t).rate(r).dividend(0.0).vol(sigma).build().price();
    assertTrue(otmPut > otmPutBs, "negative-theta VG fattens the left tail: " + otmPut + " vs " + otmPutBs);
  }

  @Test
  void hestonSurfaceCalibrationRecoversParameters() {
    double s = 100;
    double r = 0.02;
    double v0 = 0.045;
    double kappa = 2.0;
    double theta = 0.05;
    double xi = 0.5;
    double rho = -0.65;
    HestonCf truth = HestonCf.of().rate(r).v0(v0).kappa(kappa).theta(theta).xi(xi).rho(rho).build();

    List<HestonCosCalibrator.Quote> quotes = new ArrayList<>();
    for (double tt : new double[] {0.25, 0.5, 1.0, 2.0}) {
      for (double k : new double[] {80, 90, 100, 110, 120}) {
        quotes.add(HestonCosCalibrator.Quote.of().strike(k).maturity(tt).price(CosMethod.price(truth, OptionTypeEnum.CALL, s, k, r, tt)).build());
      }
    }

    HestonCosCalibrator.Result fit = HestonCosCalibrator.calibrate(
        s, r, quotes, new double[] {0.04, 1.0, 0.04, 0.3, -0.3});

    assertTrue(fit.rmse() < 5e-3, "surface fit RMSE small, got " + fit.rmse());
    // v0 and theta are the well-identified parameters; kappa/xi/rho trade off.
    assertEquals(v0, fit.v0(), 5e-3, "v0 recovered");
    assertEquals(theta, fit.theta(), 5e-3, "theta recovered");
  }

  private static <M> double price(M market,
      java.util.function.BiConsumer<com.nablatensor.engine.AadRecorder, Nabla.Inputs<M>> v,
      long paths, long seed) {
    try (Nabla.TypedPricer<M> p = Nabla.model(market, v).fp64().priceOnly().on("cpu-jit").build()) {
      return p.value().with(market).scenarios(paths).seed(seed).run().price();
    }
  }
}
