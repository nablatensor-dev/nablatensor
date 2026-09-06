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
package com.nablatensor.cva;

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.Nabla;
import com.nablatensor.engine.SDouble;
import com.nablatensor.risk.TimeProfile;
import java.util.function.BiConsumer;

/**
 * The Phase-2 exposure engine: a Monte-Carlo simulation of a netting set's
 * mark-to-market on a time grid, recorded once onto the adjoint tape.
 *
 * <p>Each path evolves a one-factor Hull-White short rate and a lognormal FX
 * spot, values every trade at every grid date from analytic {@code P(t, T)}
 * bonds, applies the CSA (variation margin with a threshold and a margin period
 * of risk), and accumulates the pathwise CVA integrand
 *
 * <pre>{@code
 * CVA = LGD * sum_k  max(V(t_k) - C(t_k), 0) * D(t_k) * ( S(t_{k-1}) - S(t_k) )
 * }</pre>
 *
 * against a piecewise-flat counterparty survival curve whose forward hazards are
 * tape inputs. One {@code .greeks()} run therefore returns the CVA <em>and</em>
 * its full risk vector — IR delta and rate vega, counterparty CS01 by tenor
 * bucket, recovery and FX — from a single reverse sweep. The bump-and-revalue
 * alternative re-runs {@link #cvaOnly} once per shocked risk factor.
 */
public final class ExposureSimulation {

  private static final double SHORT_BUCKET = 2.0;
  private static final double MID_BUCKET = 5.0;

  /**
   * Working unit for the on-tape accumulation. Mark-to-market, exposure and the
   * CVA integrand are all divided by this before they touch the tape, so the
   * forward sweep and the reverse sweep both accumulate O(1) quantities instead
   * of sums that run to 1e10 in reporting currency — the difference between a
   * stable and an unusable single-precision replay. Price, standard error and
   * gradient are multiplied back by it on the way out.
   */
  private static final double MONEY_UNIT = 1.0e6;

  private final NettingSet nettingSet;
  private final int steps;
  private final double horizon;
  private final double dt;
  private String engine = "cpu-jit";
  private Boolean fp64Override = null;

  public ExposureSimulation(NettingSet nettingSet, int steps) {
    if (steps < 2) {
      throw new IllegalArgumentException("need at least 2 grid steps, got " + steps);
    }
    this.nettingSet = nettingSet;
    this.steps = steps;
    this.horizon = nettingSet.horizonYears();
    this.dt = horizon / steps;
  }

  public ExposureSimulation on(String engine) {
    this.engine = engine;
    return this;
  }

  public String engine() {
    return engine;
  }

  /**
   * Force the replay precision. Left unset, the precision follows the engine:
   * the Vulkan engine is single-precision, every other engine runs fp64. The
   * money-scaled integrand and the {@code expm1}-form marginal-default
   * probability keep the fp32 path within Monte-Carlo error of the fp64 one.
   */
  public ExposureSimulation fp64(boolean doublePrecision) {
    this.fp64Override = doublePrecision;
    return this;
  }

  private boolean useFp64() {
    return fp64Override != null ? fp64Override : !"vulkan".equals(engine);
  }

  public int steps() {
    return steps;
  }

  public double horizonYears() {
    return horizon;
  }

  public double stepYears() {
    return dt;
  }

  public NettingSet nettingSet() {
    return nettingSet;
  }

  // ---- the recorded valuation -----------------------------------------

  private BiConsumer<AadRecorder, Nabla.Inputs<CvaMarket>> valuation(boolean emitProfile) {
    return (rec, in) -> {
      SDouble shortRate0 = in.of(CvaMarket::r0);
      SDouble level = in.of(CvaMarket::hwLevel);
      SDouble meanReversion = in.of(CvaMarket::hwMeanReversion);
      SDouble hwSigma = in.of(CvaMarket::hwSigma);
      SDouble hazardShort = in.of(CvaMarket::hazardShort);
      SDouble hazardMid = in.of(CvaMarket::hazardMid);
      SDouble hazardLong = in.of(CvaMarket::hazardLong);
      SDouble lossGivenDefault = in.of(CvaMarket::recovery).neg().add(1.0);
      SDouble fxVol = in.of(CvaMarket::fxVol);
      SDouble foreignRate = in.of(CvaMarket::fxForeignRate);

      HwShortRate model = new HwShortRate(rec, shortRate0, level, meanReversion, hwSigma, dt);
      HwShortRate.State rateState = model.start();
      SDouble fxSpot = in.of(CvaMarket::fxSpot);
      SDouble fxDrift = shortRate0.sub(foreignRate).mul(dt).sub(fxVol.mul(fxVol).mul(0.5 * dt));
      SDouble fxDiffusion = fxVol.mul(Math.sqrt(dt));

      int marginPeriodSteps = nettingSet.collateral().marginPeriodSteps(dt);
      boolean collateralised = nettingSet.collateral().isCollateralised();
      double threshold = nettingSet.collateral().threshold() / MONEY_UNIT;
      double independentAmount = nettingSet.collateral().independentAmount() / MONEY_UNIT;
      SDouble[] pastValue = new SDouble[steps + 1];

      SDouble cva = rec.constant(0.0);
      // S(t_{k-1}) carried as a running product, so the marginal default
      // probability is never formed as the difference of two nearly-equal
      // survivals (the dominant single-precision cancellation).
      SDouble survivalPrevious = rec.constant(1.0);
      double previousTime = 0.0;

      for (int k = 1; k <= steps; k++) {
        rateState = model.step(rateState, rec.randn());
        fxSpot = fxSpot.mul(fxDrift.add(fxDiffusion.mul(rec.randn())).exp());
        double tk = k * dt;

        HwShortRate.State currentRate = rateState;
        SDouble currentFxSpot = fxSpot;
        CvaTrade.Path path = new CvaTrade.Path() {
          @Override
          public AadRecorder recorder() {
            return rec;
          }

          @Override
          public HwShortRate rates() {
            return model;
          }

          @Override
          public SDouble shortRate() {
            return currentRate.rate();
          }

          @Override
          public SDouble fxSpot() {
            return currentFxSpot;
          }

          @Override
          public SDouble foreignDiscount(double from, double to) {
            return foreignRate.mul(-(to - from)).exp();
          }
        };

        SDouble value = rec.constant(0.0);
        for (CvaTrade trade : nettingSet.trades()) {
          value = value.add(trade.markToMarket(path, tk));
        }
        value = value.mul(1.0 / MONEY_UNIT); // non-dimensionalise onto the tape
        pastValue[k] = value;

        SDouble exposure = value;
        if (collateralised) {
          // collateral the counterparty had posted to us as of the margin-period lag:
          // max(V(t - MPoR) - threshold, 0). The collateralised exposure at default is
          // the residual gap max(V(t) - C - independentAmount, 0) <= max(V(t), 0).
          SDouble reference = k - marginPeriodSteps >= 1
              ? pastValue[k - marginPeriodSteps]
              : rec.constant(0.0);
          SDouble posted = reference.sub(threshold).max(0.0);
          exposure = value.sub(posted).sub(independentAmount);
        }

        SDouble positiveExposure = exposure.max(0.0);
        // Marginal default probability over (t_{k-1}, t_k] as
        //   S(t_{k-1}) * (1 - e^{-deltaLambda})
        // with deltaLambda the *incremental* cumulative hazard over the step and
        // (1 - e^{-x}) from its series, so no large survivals are subtracted.
        SDouble deltaHazard = incrementalHazard(previousTime, tk, hazardShort, hazardMid, hazardLong);
        SDouble defaultProbability = survivalPrevious.mul(oneMinusExpNeg(deltaHazard));
        SDouble discount = model.discountFactor(currentRate);
        cva = cva.add(positiveExposure.mul(discount).mul(defaultProbability).mul(lossGivenDefault));
        survivalPrevious = survivalPrevious.mul(deltaHazard.neg().exp()); // S(t_k)
        previousTime = tk;

        if (emitProfile) {
          rec.output("epe_" + k, positiveExposure);
          rec.output("ee_" + k, exposure);
        }
      }
      if (!emitProfile) {
        // a single unnamed output — every engine exposes it as the default,
        // whereas a lone named output is collapsed to "value" on some backends
        rec.output(cva);
      }
    };
  }

  /**
   * The cumulative forward hazard over {@code (from, to]} from the three
   * piecewise-flat buckets {@code [0,2y] / [2y,5y] / [5y,+)}. Building the step
   * increment directly — rather than differencing two integrals from zero —
   * is what lets the marginal default probability stay accurate in fp32.
   */
  private static SDouble incrementalHazard(double from, double to, SDouble hazardShort,
                                           SDouble hazardMid, SDouble hazardLong) {
    double shortWidth = overlap(from, to, 0.0, SHORT_BUCKET);
    double midWidth = overlap(from, to, SHORT_BUCKET, MID_BUCKET);
    double longWidth = overlap(from, to, MID_BUCKET, Double.POSITIVE_INFINITY);
    SDouble h = null;
    if (shortWidth > 0.0) {
      h = hazardShort.mul(shortWidth);
    }
    if (midWidth > 0.0) {
      h = h == null ? hazardMid.mul(midWidth) : h.add(hazardMid.mul(midWidth));
    }
    if (longWidth > 0.0) {
      h = h == null ? hazardLong.mul(longWidth) : h.add(hazardLong.mul(longWidth));
    }
    return h; // to > from guarantees at least one positive width
  }

  private static double overlap(double from, double to, double lo, double hi) {
    return Math.max(0.0, Math.min(to, hi) - Math.max(from, lo));
  }

  /**
   * {@code 1 - e^{-x}} for small non-negative {@code x}, as the Horner form of
   * its Maclaurin series {@code x - x^2/2! + x^3/3! - ...}. The direct
   * {@code 1 - exp(-x)} loses most of its significant digits to cancellation
   * when {@code x} is small; the series has none. Six terms carry it to double
   * accuracy for {@code x <= 0.5} and to better than single-precision epsilon
   * for {@code x} up to ~1 — far beyond the step hazard {@code lambda*dt}
   * (~1e-2) this is called with.
   */
  private static SDouble oneMinusExpNeg(SDouble x) {
    SDouble s = x.mul(-1.0 / 6.0).add(1.0);
    s = x.mul(s).mul(-1.0 / 5.0).add(1.0);
    s = x.mul(s).mul(-1.0 / 4.0).add(1.0);
    s = x.mul(s).mul(-1.0 / 3.0).add(1.0);
    s = x.mul(s).mul(-1.0 / 2.0).add(1.0);
    return x.mul(s);
  }

  // ---- runs ---------------------------------------------------------

  /**
   * CVA and the full CVA gradient from one adjoint sweep on {@link #engine}, plus
   * the expected-exposure profile. The CVA value and gradient come from a single
   * {@code greeks()} kernel over the scalar CVA — the timed sweep. The per-date
   * profile is a separate {@code priceOnly()} kernel with many named outputs;
   * that runs on {@link #engine} where the backend supports it and falls back to
   * {@code cpu-jit} otherwise (it is a picture, not a headline number).
   */
  public CvaResult run(CvaMarket market, long paths, long seed) {
    market.validated();
    double buildSeconds;
    double cvaValue;
    double standardError;
    CvaMarket gradient;
    double sweepSeconds;
    double scenariosPerSecond;
    long scenarios;
    String engineName;

    var model = Nabla.model(market, valuation(false));
    try (Nabla.TypedPricer<CvaMarket> pricer =
             (useFp64() ? model.fp64() : model.fp32()).greeks().on(engine).build()) {
      buildSeconds = pricer.buildSeconds() + pricer.recordSeconds();
      Nabla.TypedValuation<CvaMarket> valued =
          pricer.value().with(market).scenarios(paths).seed(seed).run();
      cvaValue = valued.price() * MONEY_UNIT;
      standardError = valued.standardError() * MONEY_UNIT;
      gradient = valued.greeks().scale(MONEY_UNIT);
      sweepSeconds = valued.seconds();
      scenariosPerSecond = valued.scenariosPerSecond();
      scenarios = valued.scenarios();
      engineName = pricer.engine();
    }

    TimeProfile[] profiles = profile(market, Math.min(paths, 50_000L), seed);
    return new CvaResult(cvaValue, standardError, market, profiles[0], profiles[1],
        gradient, sweepSeconds, buildSeconds, scenariosPerSecond, scenarios, engineName);
  }

  private TimeProfile[] profile(CvaMarket market, long paths, long seed) {
    for (String candidate : engine.equals("cpu-jit") ? new String[] {"cpu-jit"}
        : new String[] {engine, "cpu-jit"}) {
      try {
        double[] times = new double[steps];
        double[] epe = new double[steps];
        double[] ee = new double[steps];
        var model = Nabla.model(market, valuation(true));
        boolean candidateFp64 = fp64Override != null ? fp64Override : !"vulkan".equals(candidate);
        try (Nabla.TypedPricer<CvaMarket> profiler =
                 (candidateFp64 ? model.fp64() : model.fp32()).priceOnly().on(candidate).build()) {
          Nabla.Valuation raw =
              profiler.value().with(market).scenarios(paths).seed(seed).run().valuation();
          for (int k = 1; k <= steps; k++) {
            times[k - 1] = k * dt;
            epe[k - 1] = raw.price("epe_" + k) * MONEY_UNIT;
            ee[k - 1] = raw.price("ee_" + k) * MONEY_UNIT;
          }
        }
        return new TimeProfile[] {new TimeProfile(times, epe), new TimeProfile(times, ee)};
      } catch (RuntimeException unsupported) {
        if (candidate.equals("cpu-jit")) {
          throw unsupported;
        }
      }
    }
    throw new IllegalStateException("unreachable");
  }

  /** Just the CVA number — the kernel a prescribed-bump sensitivity re-runs. */
  public double cvaOnly(CvaMarket market, long paths, long seed) {
    market.validated();
    var model = Nabla.model(market, valuation(false));
    try (Nabla.TypedPricer<CvaMarket> pricer =
             (useFp64() ? model.fp64() : model.fp32()).priceOnly().on(engine).build()) {
      return pricer.value().with(market).scenarios(paths).seed(seed).run().price() * MONEY_UNIT;
    }
  }
}
