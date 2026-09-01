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
package com.nablatensor.reg.frtb.sa;

import com.nablatensor.reg.frtb.drc.DefaultRiskCharge;
import com.nablatensor.reg.frtb.drc.DefaultRiskPosition;
import com.nablatensor.reg.frtb.rrao.Rrao;
import com.nablatensor.reg.frtb.sbm.CurvatureRepricing;
import com.nablatensor.reg.frtb.sbm.SbmCharge;
import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.Sensitivities;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * The FRTB SA own-funds requirement assembler:
 *
 * <pre>{@code
 * total = Sum_class max(high, medium, low)          (SBM: 7 risk classes)
 *       + DRC                                        (default risk charge)
 *       + RRAO                                       (residual risk add-on)
 * }</pre>
 *
 * <p>SBM sensitivities are supplied as one {@link Sensitivities} vector keyed by
 * {@link com.nablatensor.risk.RiskFactor}s; the assembler dispatches each risk
 * class to its {@link com.nablatensor.risk.RiskClassProfile} from the chosen
 * {@link FrtbParameterSet}. Curvature repricings, DRC positions and RRAO
 * positions are supplied as lists.
 *
 * <p>Calculators, not sign-off.
 */
public final class FrtbSa {

  /** The assembled FRTB SA result. */
  public record Result(double sbm, double drc, double rrao, double total,
                       Map<RiskClass, SbmCharge.Result> perRiskClass,
                       DefaultRiskCharge.Result drcDetail,
                       Rrao.Result rraoDetail,
                       CorepMarketRisk corep) {
  }

  /** A relieved / unrelieved pair for the CRR3 COREP dual. */
  public record Dual(Result relieved, Result unrelieved) {
  }

  private final FrtbParameterSet params;
  private final ReportingCurrency reportingCurrency;
  private Sensitivities sbmBook = Sensitivities.empty();
  private List<CurvatureRepricing> curvature = List.of();
  private List<DefaultRiskPosition> drcPositions = List.of();
  private List<Rrao.ResidualRiskPosition> rraoPositions = List.of();

  private FrtbSa(FrtbParameterSet params, ReportingCurrency reportingCurrency) {
    this.params = params;
    this.reportingCurrency = reportingCurrency;
  }

  /** Basel MAR21 parameters, reporting currency {@code code}, no FX conversion table. */
  public static FrtbSa of(String reportingCurrencyCode) {
    return new FrtbSa(FrtbParameterSet.baselMar21(), ReportingCurrency.of(reportingCurrencyCode));
  }

  public static FrtbSa of(FrtbParameterSet params, ReportingCurrency reportingCurrency) {
    return new FrtbSa(params, reportingCurrency);
  }

  public FrtbSa sbm(Sensitivities book, List<CurvatureRepricing> curvatureRepricings) {
    this.sbmBook = book;
    this.curvature = curvatureRepricings == null ? List.of() : List.copyOf(curvatureRepricings);
    return this;
  }

  public FrtbSa drc(List<DefaultRiskPosition> positions) {
    this.drcPositions = positions == null ? List.of() : List.copyOf(positions);
    return this;
  }

  public FrtbSa rrao(List<Rrao.ResidualRiskPosition> positions) {
    this.rraoPositions = positions == null ? List.of() : List.copyOf(positions);
    return this;
  }

  public Result compute() {
    Map<RiskClass, SbmCharge.Result> perClass = new EnumMap<>(RiskClass.class);
    Map<RiskClass, CorepMarketRisk.Row> rows = new EnumMap<>(RiskClass.class);
    double sbmTotal = 0.0;

    for (RiskClass rc : RiskClass.values()) {
      boolean anyDelta = !sbmBook.ofClass(rc).isEmpty();
      boolean anyCurv = curvature.stream().anyMatch(c -> c.factor().riskClass() == rc);
      if (!anyDelta && !anyCurv) {
        continue;
      }
      SbmCharge.Result r = SbmCharge.of(params.profile(rc)).compute(sbmBook, curvature);
      perClass.put(rc, r);
      rows.put(rc, new CorepMarketRisk.Row(r.delta(), r.vega(), r.curvature(), r.total(), r.bindingScenario()));
      sbmTotal += r.total();
    }

    DefaultRiskCharge.Result drcDetail = DefaultRiskCharge.of(drcPositions).compute();
    Rrao.Result rraoDetail = Rrao.of(rraoPositions).compute();
    double total = sbmTotal + drcDetail.total() + rraoDetail.total();

    CorepMarketRisk corep = new CorepMarketRisk(reportingCurrency.code(), params.name(), rows,
        sbmTotal, drcDetail.total(), rraoDetail.total(), total);
    return new Result(sbmTotal, drcDetail.total(), rraoDetail.total(), total,
        perClass, drcDetail, rraoDetail, corep);
  }

  /**
   * Run the same book through two parameter sets — the CRR3 COREP
   * relieved / unrelieved dual, mechanically two replays with no recompile.
   */
  public static Dual dual(FrtbSa relieved, FrtbSa unrelieved) {
    return new Dual(relieved.compute(), unrelieved.compute());
  }

  /** The risk classes that carry any input in the current configuration, in SBM order. */
  public List<RiskClass> populatedRiskClasses() {
    List<RiskClass> out = new ArrayList<>();
    for (RiskClass rc : RiskClass.values()) {
      if (!sbmBook.ofClass(rc).isEmpty() || curvature.stream().anyMatch(c -> c.factor().riskClass() == rc)) {
        out.add(rc);
      }
    }
    return out;
  }
}
