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

import com.nablatensor.cva.BaCva;
import com.nablatensor.cva.BaCvaParameters;
import com.nablatensor.cva.CdsQuote;
import com.nablatensor.cva.CollateralAgreement;
import com.nablatensor.cva.CreditName;
import com.nablatensor.cva.Cva;
import com.nablatensor.cva.CvaCapital;
import com.nablatensor.cva.CvaHedge;
import com.nablatensor.cva.CvaMarket;
import com.nablatensor.cva.CvaResult;
import com.nablatensor.cva.CvaRiskFactors;
import com.nablatensor.cva.ExposureSimulation;
import com.nablatensor.cva.FxForward;
import com.nablatensor.cva.HazardCurve;
import com.nablatensor.cva.InterestRateSwap;
import com.nablatensor.cva.NettingSet;
import com.nablatensor.cva.PraCvaMethods;
import com.nablatensor.cva.SaCva;
import com.nablatensor.cva.SaCvaParameters;
import com.nablatensor.cva.SaCvaResult;
import com.nablatensor.cva.SaCvaSensitivities;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.Sensitivities;
import com.nablatensor.risk.TimeProfile;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * A compact, end-to-end SA-CVA / BA-CVA demonstration built around the one
 * expensive stage of a CVA capital run: the Monte-Carlo netting-set exposure
 * simulation, and its re-run once per risk factor when SA-CVA sensitivities are
 * taken by prescribed bump.
 *
 * <p>The narrated {@code demo/cva-capital.sh} walks the eight stages on
 * whichever backend is fastest here; this class is the non-narrated runner and
 * the data provider it shares. Every number is produced on the machine you run
 * it on.
 *
 * <p><b>Calculators, not sign-off.</b> Parameter tables are indicative demo
 * values; model validation, parameter attestation and regulatory submission stay
 * with the user.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.CvaShowcase}. System properties:
 * {@code -Dengine=}, {@code -Dpaths=}, {@code -Dsteps=}, {@code -Dseed=}.
 */
public final class CvaShowcase {

  private static final String PARAMETER_VERSION = "DEMO-MAR50-2026-09";
  private static final String REPORTING_CURRENCY = "USD";
  private static final String FX_PAIR = "EURUSD";

  private CvaShowcase() {
  }

  // ---- shared demo inputs -------------------------------------------

  public static CvaMarket market() {
    return CvaMarket.demo();
  }

  /** Counterparty A: a BBB financial, CDS curve quoted 90 / 130 / 150 / 170 bp. */
  public static CreditName counterpartyA() {
    HazardCurve curve = HazardCurve.bootstrap(List.of(
        new CdsQuote(1.0, 90.0), new CdsQuote(3.0, 130.0),
        new CdsQuote(5.0, 150.0), new CdsQuote(10.0, 170.0)),
        0.40, t -> Math.exp(-0.03 * t));
    return new CreditName("CPTY-A", curve, 0.40, CreditName.Rating.BBB, CreditName.Sector.FINANCIAL);
  }

  /** Counterparty B: an A-rated corporate, flat 110 bp, under a daily-margined CSA. */
  public static CreditName counterpartyB() {
    return new CreditName("CPTY-B", HazardCurve.fromFlatSpread(110.0, 0.40, 10.0),
        0.40, CreditName.Rating.A, CreditName.Sector.CORPORATE);
  }

  /** Uncollateralised netting set with A: an in-the-money payer, an offsetting
   *  receiver, and a bought EUR forward. The 7y payer keeps exposure alive in
   *  every credit-spread tenor bucket. */
  public static NettingSet nettingSetA() {
    return new NettingSet("NS-CPTY-A", counterpartyA(), List.of(
        InterestRateSwap.payer("A-SWAP-PAY", 100_000_000.0, 0.020, 7.0),
        InterestRateSwap.receiver("A-SWAP-REC", 40_000_000.0, 0.036, 5.0),
        new FxForward("A-FX-FWD", FxForward.Side.BUY_FOREIGN, 20_000_000.0, 1.05, 4.0)));
  }

  /** Daily-margined netting set with B: one in-the-money payer swap. */
  public static NettingSet nettingSetB() {
    return new NettingSet("NS-CPTY-B", counterpartyB(),
        List.of(InterestRateSwap.payer("B-SWAP-PAY", 75_000_000.0, 0.022, 7.0)),
        CollateralAgreement.dailyMargined(2_000_000.0));
  }

  public static CvaRiskFactors riskFactorsFor(NettingSet nettingSet) {
    return new CvaRiskFactors(REPORTING_CURRENCY, nettingSet.counterparty(), FX_PAIR);
  }

  /** One single-name CDS on A, notional 8m, 7y, recognised at r_hc = 1. */
  public static CvaHedge hedgeOnA() {
    return CvaHedge.singleName("CPTY-A", 8_000_000.0, 7.0, 0.05, 1.0);
  }

  // ---- non-narrated runner ----------------------------------------

  public static void main(String[] args) {
    String engine = System.getProperty("engine", "cpu-jit");
    long paths = Long.getLong("paths", 200_000L);
    int steps = Integer.getInteger("steps", 24);
    long seed = Long.getLong("seed", 42L);

    NettingSet a = nettingSetA();
    NettingSet b = nettingSetB();
    CvaMarket market = market();

    System.out.printf(Locale.ROOT, "CVA CAPITAL  |  %s  |  engine %s  |  %,d paths x %d steps%n%n",
        PARAMETER_VERSION, engine, paths, steps);

    System.out.println("[1/8] NETTING SETS AND COUNTERPARTY CURVES");
    for (NettingSet set : List.of(a, b)) {
      System.out.printf(Locale.ROOT, "      %-11s vs %-7s  %d trades  gross $%.0fm  %s%n",
          set.id(), set.counterparty().id(), set.trades().size(),
          set.grossNotional() / 1e6,
          set.collateral().isCollateralised() ? "daily-margined CSA" : "uncollateralised");
    }
    double[] survivalA = new double[] {
        a.counterparty().curve().survival(1.0), a.counterparty().curve().survival(5.0)};
    System.out.printf(Locale.ROOT, "      CPTY-A survival  1y %.4f   5y %.4f%n%n", survivalA[0], survivalA[1]);

    System.out.println("[2/8] EXPOSURE SIMULATION + ONE ADJOINT SWEEP (per netting set)");
    ExposureSimulation simA = new ExposureSimulation(a, steps).on(engine);
    CvaResult resultA = simA.run(market, paths, seed);
    printProfile(resultA.epeProfile());
    System.out.printf(Locale.ROOT,
        "      CVA(A) = %s   sweep %.3f s (value + full gradient)   build %.3f s%n",
        money(resultA.value()), resultA.sweepSeconds(), resultA.buildSeconds());

    System.out.println("\n[3/8] THE CVA RISK VECTOR, FROM THAT ONE SWEEP");
    CvaMarket g = resultA.gradient();
    double lgd = a.counterparty().lossGivenDefault();
    System.out.printf(Locale.ROOT, "      IR delta   1bp curve shift        %s%n",
        dollars((g.r0() + g.hwLevel()) * 1.0e-4));
    System.out.printf(Locale.ROOT, "      IR vega    dCVA/dsigma . sigma     %s%n",
        dollars(g.hwSigma() * resultA.market().hwSigma()));
    System.out.printf(Locale.ROOT, "      CS01       1bp spread, s / m / l   %s / %s / %s%n",
        dollars(g.hazardShort() / lgd * 1.0e-4), dollars(g.hazardMid() / lgd * 1.0e-4),
        dollars(g.hazardLong() / lgd * 1.0e-4));
    System.out.printf(Locale.ROOT, "      FX delta   1%% spot move             %s%n",
        dollars(g.fxSpot() * resultA.market().fxSpot() * 0.01));
    System.out.printf(Locale.ROOT, "      recovery   dCVA/dR                 %s%n", dollars(g.recovery()));

    System.out.println("\n[4/8] THE HEAVY ALTERNATIVE: SA-CVA SENSITIVITIES BY PRESCRIBED BUMP");
    SaCvaSensitivities.BumpResult bump = SaCvaSensitivities.bumpAndRevalue(
        simA, market, paths, seed, riskFactorsFor(a));
    System.out.printf(Locale.ROOT,
        "      %d full exposure re-simulations   %.3f s   vs one %.3f s sweep%n",
        bump.revaluations(), bump.seconds(), resultA.sweepSeconds());

    System.out.println("\n[5/8] RECONCILE: one sweep vs " + bump.revaluations() + " re-simulations");
    Sensitivities adjoint = SaCvaSensitivities.adjoint(resultA, riskFactorsFor(a));
    for (Map.Entry<RiskFactor, Double> entry : bump.sensitivities().asMap().entrySet()) {
      double bumped = entry.getValue();
      double swept = adjoint.get(entry.getKey());
      System.out.printf(Locale.ROOT, "      %-34s bump %+12.2f   sweep %+12.2f%n",
          shortName(entry.getKey()), bumped, swept);
    }

    System.out.println("\n[6/8] PORTFOLIO: BA-CVA AND SA-CVA");
    CvaCapital capital = Cva.of(market)
        .add(a, riskFactorsFor(a))
        .add(b, riskFactorsFor(b))
        .hedge(hedgeOnA())
        .steps(steps).paths(paths).seed(seed).on(engine)
        .compute();
    System.out.printf(Locale.ROOT, "      portfolio CVA        %s%n", money(capital.cvaValue()));
    System.out.printf(Locale.ROOT, "      BA-CVA reduced       %s%n", money(capital.baCva().reduced()));
    System.out.printf(Locale.ROOT, "      BA-CVA full (hedged) %s   hedge benefit %s%n",
        money(capital.baCva().full()), money(capital.baCva().hedgeBenefit()));

    SaCvaResult sa = capital.saCva();
    System.out.printf(Locale.ROOT, "      SA-CVA  LOW %s  MEDIUM %s  HIGH %s  ->  %s  %s%n",
        money(sa.perScenario().get(com.nablatensor.risk.CorrelationScenario.LOW)),
        money(sa.perScenario().get(com.nablatensor.risk.CorrelationScenario.MEDIUM)),
        money(sa.perScenario().get(com.nablatensor.risk.CorrelationScenario.HIGH)),
        sa.selected(), money(sa.total()));

    System.out.println("\n[7/8] THE THREE PRA STANDARDISED METHODS");
    PraCvaMethods methods = PraCvaMethods.of(capital);
    System.out.printf(Locale.ROOT, "      Alternative Approach %s%n", money(methods.alternativeApproach()));
    System.out.printf(Locale.ROOT, "      Basic Approach       %s%n", money(methods.basicApproach()));
    System.out.printf(Locale.ROOT, "      Standardised (SA-CVA)%s%n", money(methods.standardisedApproach()));

    System.out.println("\n[8/8] CAPITAL AND SIGN-OFF");
    System.out.printf(Locale.ROOT, "      binding CVA capital  %s  (%s)%n",
        money(methods.bindingCharge()),
        methods.bindingCharge() == methods.standardisedApproach() ? "SA-CVA"
            : methods.bindingCharge() == methods.basicApproach() ? "BA-CVA full" : "Alternative");
    System.out.printf(Locale.ROOT,
        "      total adjoint work %.3f s across %d netting sets; everything else is arithmetic.%n",
        capital.sweepSeconds(), capital.perNettingSet().size());
    System.out.println("      Educational evidence package; not a regulatory filing.");
  }

  private static void printProfile(TimeProfile epe) {
    int width = 40;
    double peak = 0.0;
    for (double v : epe.values()) {
      peak = Math.max(peak, v);
    }
    for (int i = 0; i < epe.values().length; i += Math.max(1, epe.values().length / 12)) {
      int bars = peak > 0.0 ? (int) Math.round(width * epe.values()[i] / peak) : 0;
      System.out.printf(Locale.ROOT, "      t=%4.2fy  %s %s%n",
          epe.times()[i], "#".repeat(bars) + " ".repeat(width - bars), money(epe.values()[i]));
    }
  }

  private static String money(double value) {
    double millions = value / 1e6;
    return String.format(Locale.ROOT, "$%,.3fm", millions);
  }

  private static String dollars(double value) {
    return String.format(Locale.ROOT, "$%,.0f", value);
  }

  private static String shortName(RiskFactor factor) {
    return factor.riskClass() + " " + factor.measure() + " "
        + factor.bucket() + "/" + factor.name()
        + (factor.tenor() > 0.0 ? " @" + factor.tenor() + "y" : "");
  }
}
