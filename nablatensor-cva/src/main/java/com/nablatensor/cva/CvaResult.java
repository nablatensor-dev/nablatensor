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

import com.nablatensor.codegen.Of;

import com.nablatensor.risk.TimeProfile;

/**
 * The output of one {@link ExposureSimulation} run: the unilateral CVA, its
 * Monte-Carlo standard error, the expected-exposure profiles, the full CVA
 * gradient from a single adjoint sweep, and timing.
 *
 * @param value               unilateral CVA in reporting currency
 * @param standardError       Monte-Carlo standard error of {@link #value}
 * @param market              the market the CVA was valued at
 * @param epeProfile          expected positive exposure {@code E[max(V(t), 0)]} by grid date
 * @param eeProfile           expected exposure {@code E[V(t)]} (signed) by grid date
 * @param gradient            {@code dCVA/d(input)} for every {@link CvaMarket} component, one sweep
 * @param sweepSeconds        wall time of the adjoint run (value + full gradient)
 * @param buildSeconds        wall time to record and compile the tape
 * @param scenariosPerSecond  Monte-Carlo throughput
 * @param scenarios           path count
 * @param engine              the engine the tape ran on
 */
@Of
public final class CvaResult {

  private final double value;
  private final double standardError;
  private final CvaMarket market;
  private final TimeProfile epeProfile;
  private final TimeProfile eeProfile;
  private final CvaMarket gradient;
  private final double sweepSeconds;
  private final double buildSeconds;
  private final double scenariosPerSecond;
  private final long scenarios;
  private final String engine;

  private CvaResult(double value, double standardError, CvaMarket market, TimeProfile epeProfile, TimeProfile eeProfile, CvaMarket gradient, double sweepSeconds, double buildSeconds, double scenariosPerSecond, long scenarios, String engine) {
    this.value = value;
    this.standardError = standardError;
    this.market = market;
    this.epeProfile = epeProfile;
    this.eeProfile = eeProfile;
    this.gradient = gradient;
    this.sweepSeconds = sweepSeconds;
    this.buildSeconds = buildSeconds;
    this.scenariosPerSecond = scenariosPerSecond;
    this.scenarios = scenarios;
    this.engine = engine;
  }

  static CvaResult create(double value, double standardError, CvaMarket market, TimeProfile epeProfile, TimeProfile eeProfile, CvaMarket gradient, double sweepSeconds, double buildSeconds, double scenariosPerSecond, long scenarios, String engine) {
    return new CvaResult(value, standardError, market, epeProfile, eeProfile, gradient, sweepSeconds, buildSeconds, scenariosPerSecond, scenarios, engine);
  }

  public static CvaResultBuilder of() {
    return new CvaResultBuilder();
  }

  public double value() {
    return value;
  }

  public double standardError() {
    return standardError;
  }

  public CvaMarket market() {
    return market;
  }

  public TimeProfile epeProfile() {
    return epeProfile;
  }

  public TimeProfile eeProfile() {
    return eeProfile;
  }

  public CvaMarket gradient() {
    return gradient;
  }

  public double sweepSeconds() {
    return sweepSeconds;
  }

  public double buildSeconds() {
    return buildSeconds;
  }

  public double scenariosPerSecond() {
    return scenariosPerSecond;
  }

  public long scenarios() {
    return scenarios;
  }

  public String engine() {
    return engine;
  }


  /** Plain time-average expected positive exposure over the profile. */
  public double expectedPositiveExposure() {
    return epeProfile.epe();
  }

  /** Peak expected positive exposure across the profile. */
  public double peakExpectedExposure() {
    double peak = 0.0;
    for (double v : epeProfile.values()) {
      peak = Math.max(peak, v);
    }
    return peak;
  }

  /**
   * Basel effective EPE: the time-average of the non-decreasing envelope of the
   * EPE profile over {@code [0, min(1yr, horizon)]}.
   */
  public double effectiveExpectedPositiveExposure() {
    double[] times = epeProfile.times();
    double[] epe = epeProfile.values();
    double horizon = Math.min(1.0, times[times.length - 1]);
    double runningMax = 0.0;
    double weighted = 0.0;
    double previousTime = 0.0;
    int counted = 0;
    for (int i = 0; i < times.length && previousTime < horizon + 1.0e-9; i++) {
      runningMax = Math.max(runningMax, epe[i]);
      double dt = times[i] - previousTime;
      weighted += runningMax * dt;
      previousTime = times[i];
      counted++;
    }
    return counted == 0 ? 0.0 : weighted / Math.min(horizon, previousTime);
  }
}
