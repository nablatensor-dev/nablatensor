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

import com.nablatensor.risk.Sensitivities;
import java.util.ArrayList;
import java.util.List;

/**
 * The Phase-2 assembler: from a set of netting sets and a market, run the
 * exposure simulation per netting set (one adjoint sweep each), net the CVA
 * sensitivities, and compute both the BA-CVA and SA-CVA capital charges.
 *
 * <pre>{@code
 * CvaCapital capital = Cva.of(market)
 *     .add(nettingSetA, riskFactorsA)
 *     .add(nettingSetB, riskFactorsB)
 *     .hedge(CvaHedge.singleName("CPTY-A", 25e6, 5.0, 0.05, 1.0))
 *     .paths(500_000).steps(24).on("vulkan")
 *     .compute();
 * }</pre>
 *
 * <p><b>Calculators, not sign-off.</b> This computes the numbers MAR50 / CRR3
 * Articles 383-384 ask for. Model validation, parameter attestation and
 * regulatory submission stay with the user; the parameter tables in
 * {@link SaCvaParameters} and {@link BaCvaParameters} are indicative demo values.
 */
public final class Cva {

  private final CvaMarket market;
  private final List<NettingSet> nettingSets = new ArrayList<>();
  private final List<CvaRiskFactors> riskFactors = new ArrayList<>();
  private final List<CvaHedge> hedges = new ArrayList<>();

  private int steps = 24;
  private long paths = 200_000L;
  private long seed = 0x5DEECE66DL;
  private String engine = "cpu-jit";
  private SaCvaParameters saCvaParameters = SaCvaParameters.demo();
  private BaCvaParameters baCvaParameters = BaCvaParameters.standard();

  private Cva(CvaMarket market) {
    this.market = market.validated();
  }

  public static Cva of(CvaMarket market) {
    return new Cva(market);
  }

  public Cva add(NettingSet nettingSet, CvaRiskFactors keys) {
    nettingSets.add(nettingSet);
    riskFactors.add(keys);
    return this;
  }

  public Cva hedge(CvaHedge hedge) {
    hedges.add(hedge);
    return this;
  }

  public Cva steps(int steps) {
    this.steps = steps;
    return this;
  }

  public Cva paths(long paths) {
    this.paths = paths;
    return this;
  }

  public Cva seed(long seed) {
    this.seed = seed;
    return this;
  }

  public Cva on(String engine) {
    this.engine = engine;
    return this;
  }

  public Cva saCvaParameters(SaCvaParameters parameters) {
    this.saCvaParameters = parameters;
    return this;
  }

  public Cva baCvaParameters(BaCvaParameters parameters) {
    this.baCvaParameters = parameters;
    return this;
  }

  public CvaCapital compute() {
    if (nettingSets.isEmpty()) {
      throw new IllegalStateException("add at least one netting set");
    }
    List<CvaResult> results = new ArrayList<>();
    List<BaCva.Exposure> exposures = new ArrayList<>();
    Sensitivities netted = Sensitivities.empty();
    double cvaValue = 0.0;
    double sweepSeconds = 0.0;
    int bumpRevaluationsAvoided = 0;

    for (int i = 0; i < nettingSets.size(); i++) {
      NettingSet nettingSet = nettingSets.get(i);
      CvaRiskFactors keys = riskFactors.get(i);
      ExposureSimulation simulation = new ExposureSimulation(nettingSet, steps).on(engine);
      CvaResult result = simulation.run(market, paths, seed + i);
      results.add(result);
      cvaValue += result.value();
      sweepSeconds += result.sweepSeconds();

      netted = netted.plus(SaCvaSensitivities.adjoint(result, keys));
      bumpRevaluationsAvoided += 2 * (5 + CvaRiskFactors.creditSpreadVertexCount());

      // Demo EAD proxy: alpha * time-average expected positive exposure over the
      // netting set's life. A production run would feed the SA-CCR or IMM EAD.
      exposures.add(new BaCva.Exposure(nettingSet.counterparty(),
          nettingSet.effectiveMaturityYears(),
          baCvaParameters.alpha() * result.expectedPositiveExposure()));
    }

    BaCvaResult baCva = new BaCva(baCvaParameters).charge(exposures, hedges);
    SaCvaResult saCva = new SaCva(saCvaParameters).charge(netted);

    return new CvaCapital(cvaValue, results, netted, baCva, saCva,
        sweepSeconds, bumpRevaluationsAvoided);
  }
}
