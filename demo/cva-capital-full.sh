#!/usr/bin/env bash
# CVA capital in full: the whole standardised CVA regulation, written out stage
# by stage — the counterparty curves, the exposure model, SA-CVA sensitivities
# and aggregation, BA-CVA reduced and full, and the three PRA methods.
#   ./demo/cva-capital-full.sh [--fast] [--cpu]
# The exposure integrand is non-dimensionalised and the marginal default
# probability is evaluated in expm1 form, so the single-precision replay holds
# to Monte-Carlo error. This runs on the fastest adjoint backend available
# (CUDA > Vulkan > cpu-jit); override with NABLATENSOR_DEMO_ENGINE=cuda|vulkan|cpu-jit.
DEMO_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_player.sh"

player_start "CVA capital in full, on the fastest backend" \
             "every stage of the standardised CVA charge, in code"

note "fastest adjoint backend here: $(capture <<'PROBE'
System.out.print(ENGINE + "  (" + ENGINE_DESC + ")");
PROBE
)"

quiet <<'SETUP'
import com.nablatensor.cva.*;
import com.nablatensor.risk.*;
import com.nablatensor.examples.CvaShowcase;
long N = Long.getLong("nablatensor.demo.paths", 200_000L);
long SEED = 42L;
int STEPS = 28;
String orange(Object s) { return color("38;5;208", s); }
String money(double v) { return String.format(Locale.ROOT, "$%,.3fm", v / 1e6); }
String dol(double v) { return String.format(Locale.ROOT, "$%,.0f", v); }
void bar(TimeProfile p) {
  double peak = 0; for (double v : p.values()) peak = Math.max(peak, v);
  for (int i = 0; i < p.values().length; i += 2) {
    int n = (int) Math.round(34 * p.values()[i] / peak);
    System.out.printf(Locale.ROOT, "   %4.2fy  %s%s  %s%n", p.times()[i],
        "#".repeat(n), " ".repeat(34 - n), money(p.values()[i]));
  }
}
void row(String k, double v) {
  System.out.printf(Locale.ROOT, "   %-24s %s%n", k, dol(v));
}
SETUP

banner "1 · The portfolio, the market, the rulebook"
say "Two master agreements: an uncollateralised book with a BBB financial, and"
say "a daily-margined book with an A-rated corporate. One CVA market snapshot,"
say "one versioned SA-CVA / BA-CVA parameter set."

run <<'CODE'
var mkt = CvaShowcase.market();
var nsA = CvaShowcase.nettingSetA();
var nsB = CvaShowcase.nettingSetB();
var keysA = CvaShowcase.riskFactorsFor(nsA);
for (var t : nsA.trades()) System.out.println(grey("   A  " + t));
for (var t : nsB.trades()) System.out.println(grey("   B  " + t));
CODE

run <<'CODE'
for (var ns : new NettingSet[]{nsA, nsB})
  System.out.printf(Locale.ROOT, "   %-11s vs %-7s  gross %s  horizon %.0fy  %s%n",
      ns.id(), ns.counterparty().id(), money(ns.grossNotional()), ns.horizonYears(),
      ns.collateral().isCollateralised() ? "daily-margined CSA" : "uncollateralised");
CODE

banner "2 · Counterparty curves: CDS quotes -> hazard -> survival"
say "A's par CDS quotes (90 / 130 / 150 / 170 bp) are stripped to a piecewise-"
say "flat forward hazard; B is a flat 110 bp. Survival is exp(-integral of the"
say "hazard), and its drop over a step is that step's marginal default"
say "probability."

run <<'CODE'
var curveA = nsA.counterparty().curve();
double[] kt = curveA.knotTimes();
double[] fh = curveA.forwardHazards();
for (int i = 0; i < kt.length; i++)
  System.out.printf(Locale.ROOT, "   segment to %4.1fy   forward hazard %.4f%n", kt[i], fh[i]);
CODE

run <<'CODE'
for (double t : new double[]{1,3,5,7,10})
  System.out.printf(Locale.ROOT, "   S(%2.0fy) = %.4f    dPD(%2.0f->%2.0f) = %.4f%n",
      t, curveA.survival(t), t, t + 1, curveA.defaultProbability(t, t + 1));
CODE

banner "3 · The exposure model"
say "Each path evolves a one-factor Hull-White short rate and a lognormal FX"
say "spot on the adjoint tape, values every trade at every step from analytic"
say "bonds, nets inside the master agreement, and applies variation margin"
say "with a threshold and a margin period of risk."

run <<'CODE'
var csaB = nsB.collateral();
System.out.printf(Locale.ROOT, "   A  %s%n",
    nsA.collateral().isCollateralised() ? "CSA"
        : "no CSA — exposure is the netted MtM when positive, else zero");
System.out.printf(Locale.ROOT, "   B  threshold %s   IA %s   MPoR %.0f calendar days%n",
    money(csaB.threshold()), money(csaB.independentAmount()),
    csaB.marginPeriodOfRiskDays());
CODE

banner "4 · THE HEAVY PART: Monte-Carlo netting-set exposure simulation"
note "one compiled kernel, replayed over every path — the only costly stage"

run <<'CODE'
var simA = new ExposureSimulation(nsA, STEPS).on(ENGINE);
var resA = simA.run(mkt, N, SEED);
bar(resA.epeProfile());
CODE

run <<'CODE'
System.out.printf(Locale.ROOT, "   time-average EPE %s   peak EE %s%n",
    money(resA.expectedPositiveExposure()), money(resA.peakExpectedExposure()));
String se = Double.isNaN(resA.standardError()) ? "" : "  +/- " + money(resA.standardError());
System.out.printf(Locale.ROOT, "   CVA(A) = %s%s   (%,d paths x %d steps, %.3f s)%n",
    green(money(resA.value())), se, resA.scenarios(), STEPS, resA.sweepSeconds());
CODE

run <<'CODE'
System.out.println(orange("   // cost shape: N_paths x N_steps x trades x per-trade pricing"));
System.out.println(orange("   // SA-CVA needs dCVA/d(risk factor) for every prescribed factor;"));
System.out.println(orange("   // by bump-and-revalue that is this whole simulation, once per factor."));
CODE

banner "5 · The whole CVA risk vector, from ONE reverse sweep"
say "The recorded valuation differentiated once: rates, credit-spread, FX and"
say "recovery sensitivities all fall out of the single sweep that produced the"
say "number above — no re-simulation."

run <<'CODE'
var g = resA.gradient();
double lgd = nsA.counterparty().lossGivenDefault();
row("IR delta   (1bp)", (g.r0() + g.hwLevel()) * 1e-4);
row("IR vega", g.hwSigma() * mkt.hwSigma());
row("CS01 short (1bp)", g.hazardShort() / lgd * 1e-4);
row("CS01 mid   (1bp)", g.hazardMid() / lgd * 1e-4);
row("CS01 long  (1bp)", g.hazardLong() / lgd * 1e-4);
row("FX delta   (1%)", g.fxSpot() * mkt.fxSpot() * 0.01);
row("recovery   (dCVA/dR)", g.recovery());
CODE

run <<'CODE'
System.out.printf(Locale.ROOT, "   sweep %s s   build %s s%n",
    yellow(String.format("%.3f", resA.sweepSeconds())),
    String.format("%.3f", resA.buildSeconds()));
CODE

banner "6 · The letter-compliant alternative: bump every factor, re-simulate"
note "the SA-CVA sensitivity as prescribed: one full exposure re-run per shock"

run <<'CODE'
var bump = SaCvaSensitivities.bumpAndRevalue(simA, mkt, N, SEED, keysA);
System.out.printf(Locale.ROOT, "   %d full re-simulations in %s s   vs one %s s sweep%n",
    bump.revaluations(), yellow(String.format("%.2f", bump.seconds())),
    yellow(String.format("%.2f", resA.sweepSeconds())));
CODE

run <<'CODE'
System.out.println(orange("   // a production book carries 100+ prescribed factors —"));
System.out.println(orange("   // the exposure Monte-Carlo is where the run time goes."));
CODE

banner "7 · Reconcile: one sweep vs every re-simulation"
say "Same seed, same paths, same tape — the two agree to the bump's own"
say "second-order error."

run <<'CODE'
var adj = SaCvaSensitivities.adjoint(resA, keysA);
for (var e : bump.sensitivities().asMap().entrySet()) {
  var k = e.getKey();
  String tag = k.riskClass() + " " + k.measure() + " b" + k.bucket()
      + (k.tenor() > 0 ? " @" + k.tenor() + "y" : "");
  System.out.printf(Locale.ROOT, "   %-28s bump %+11.2f   sweep %+11.2f%n",
      tag, e.getValue(), adj.get(k));
}
CODE

banner "8 · SA-CVA: weight, correlate, LOW / MEDIUM / HIGH"
say "The netted sensitivities are risk-weighted and aggregated FRTB-style —"
say "within bucket, then across buckets — under three correlation scenarios."
say "m_CVA scales the result; the binding scenario is the worst of the three."

run <<'CODE'
var bld = Cva.of(mkt).add(nsA, keysA).add(nsB, CvaShowcase.riskFactorsFor(nsB));
var cap = bld.hedge(CvaShowcase.hedgeOnA()).steps(STEPS).paths(N).seed(SEED).on(ENGINE).compute();
var sa = cap.saCva();
CODE

run <<'CODE'
for (var rt : sa.byRiskType().entrySet())
  System.out.printf(Locale.ROOT, "   %-14s %s%n", rt.getKey(), money(rt.getValue()));
System.out.printf(Locale.ROOT, "   SA-CVA   L %s   M %s   H %s   ->  %s  %s%n",
    money(sa.perScenario().get(CorrelationScenario.LOW)),
    money(sa.perScenario().get(CorrelationScenario.MEDIUM)),
    money(sa.perScenario().get(CorrelationScenario.HIGH)),
    sa.selected(), green(money(sa.total())));
CODE

banner "9 · BA-CVA reduced: the closed form"
say "No simulation: a supervisory risk weight per counterparty on the effective"
say "EAD, combined with one systematic factor (rho = 0.5) and an idiosyncratic"
say "remainder. This is the whole calculation."

run <<'CODE'
for (var e : cap.scvaByCounterparty().entrySet())
  System.out.printf(Locale.ROOT, "   SCVA  %-8s %s%n", e.getKey(), money(e.getValue()));
System.out.printf(Locale.ROOT, "   BA-CVA reduced = %s%n", green(money(cap.baCva().reduced())));
CODE

banner "10 · BA-CVA full: CDS hedge recognition"
say "The full version blends the unhedged charge (beta = 0.25) with a hedged"
say "one that nets eligible single-name and index CDS against each SCVA, less"
say "an indirect-hedge misalignment penalty."

run <<'CODE'
var pstd = BaCvaParameters.standard();
System.out.printf(Locale.ROOT, "   params  rho %.2f   beta %.2f   alpha %.2f%n",
    pstd.rho(), pstd.beta(), pstd.alpha());
System.out.println("   hedge   single-name CDS on CPTY-A, 8m notional, 7y");
System.out.printf(Locale.ROOT, "   BA-CVA full = %s   reduced = %s   hedge benefit %s%n",
    green(money(cap.baCva().full())), money(cap.baCva().reduced()),
    money(cap.baCva().hedgeBenefit()));
CODE

banner "11 · The three PRA standardised methods, and the binding charge"
say "UK Basel 3.1 removes the CVA internal model. A bank uses the Alternative"
say "Approach, the Basic Approach, or — with approval and a CVA desk — SA-CVA."

run <<'CODE'
var pra = PraCvaMethods.of(cap);
System.out.printf(Locale.ROOT, "   Alternative Approach    %s%n", money(pra.alternativeApproach()));
System.out.printf(Locale.ROOT, "   Basic Approach          %s%n", money(pra.basicApproach()));
System.out.printf(Locale.ROOT, "   Standardised (SA-CVA)   %s%n", money(pra.standardisedApproach()));
CODE

run <<'CODE'
double kbind = pra.bindingCharge();
String which = "Alternative Approach";
if (kbind == pra.basicApproach()) { which = "Basic Approach"; }
if (kbind == pra.standardisedApproach()) { which = "SA-CVA"; }
System.out.printf(Locale.ROOT, "   binding CVA capital     %s  (%s)%n", green(money(kbind)), which);
System.out.printf(Locale.ROOT, "   CVA RWA  (x 12.5)       %s%n", money(kbind * 12.5));
CODE

banner "12 · Where the time went"
say "One adjoint sweep per netting set produced every CVA sensitivity above."
say "The bump route re-ran the exposure simulation once per prescribed factor;"
say "SA-CVA aggregation, BA-CVA and the PRA methods are arithmetic on top."

run <<'CODE'
System.out.printf(Locale.ROOT, "   adjoint work %.2f s across %d netting sets%n",
    cap.sweepSeconds(), cap.perNettingSet().size());
System.out.println(grey("   Calculators, not sign-off — model validation and filing stay with you."));
CODE

finale "The netting-set exposure simulation was the only expensive stage." \
       "One sweep gave every SA-CVA sensitivity; the bump route re-ran it per factor." \
       "SA-CVA, BA-CVA reduced and full, and the three PRA methods, end to end."
