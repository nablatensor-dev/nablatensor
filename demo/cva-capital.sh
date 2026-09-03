#!/usr/bin/env bash
# CVA capital: a netting-set exposure simulation, then the whole SA-CVA risk
# vector from one adjoint sweep instead of one re-simulation per risk factor.
#   ./demo/cva-capital.sh [--fast] [--cpu]
# The CVA integrand is accumulated in fp64, so this runs on the fastest
# fp64-capable adjoint backend (CUDA > ROCm > cpu-jit — never the fp32-only
# Vulkan engine); override with NABLATENSOR_DEMO_ENGINE=cuda|rocm|cpu-jit.
DEMO_ARGS=("$@")
export NABLATENSOR_DEMO_FP64=1
source "$(dirname "${BASH_SOURCE[0]}")/_player.sh"

player_start "CVA capital on the fastest backend" \
             "one adjoint sweep replaces one re-simulation per risk factor"

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
String money(double v) { return String.format(Locale.ROOT, "$%,.3fm", v / 1e6); }
String dol(double v) { return String.format(Locale.ROOT, "$%,.0f", v); }
void bar(TimeProfile p) {
  double peak = 0; for (double v : p.values()) peak = Math.max(peak, v);
  for (int i = 0; i < p.values().length; i += 2) {
    int n = (int) Math.round(38 * p.values()[i] / peak);
    System.out.printf(Locale.ROOT, "   %4.2fy  %s%s  %s%n", p.times()[i],
        "#".repeat(n), " ".repeat(38 - n), money(p.values()[i]));
  }
}
SETUP

banner "1 · The netting set, the market, the counterparty"
say "One master agreement with a BBB financial: an in-the-money payer swap, an"
say "offsetting receiver, and a bought EUR forward. The counterparty is quoted"
say "in CDS; recovery 40%."

run <<'CODE'
var ns = CvaShowcase.nettingSetA();
var mkt = CvaShowcase.market();
var keys = CvaShowcase.riskFactorsFor(ns);
for (var t : ns.trades()) System.out.println(grey("   " + t));
CODE

run <<'CODE'
System.out.printf(Locale.ROOT, "   %s vs %s   gross %s   horizon %.0fy%n",
    ns.id(), ns.counterparty().id(), money(ns.grossNotional()), ns.horizonYears());
CODE

banner "2 · The counterparty curve: CDS quotes -> hazard -> survival"
say "The par quotes are stripped to a piecewise-flat forward hazard. Survival"
say "is exp(-integral of the hazard); the drop over a step is that step's"
say "default probability."

run <<'CODE'
var curve = ns.counterparty().curve();
for (double t : new double[]{1,3,5,7,10})
  System.out.printf(Locale.ROOT, "   S(%2.0fy) = %.4f     lambda(%2.0fy) = %.4f%n",
      t, curve.survival(t), t, curve.hazardAt(t));
CODE

banner "3 · Simulate the netting-set exposure on the fastest fp64 backend"
say "Each path evolves a Hull-White short rate and an FX spot, values every"
say "trade at every step from analytic bonds, nets them, and accumulates the"
say "pathwise CVA integrand against the survival curve."

run <<'CODE'
var sim = new ExposureSimulation(ns, STEPS).on(ENGINE);
var res = sim.run(mkt, N, SEED);
bar(res.epeProfile());
CODE

run <<'CODE'
String se = Double.isNaN(res.standardError()) ? "" : "  +/- " + money(res.standardError());
System.out.printf(Locale.ROOT, "   CVA = %s%s   (%,d paths, %d steps)%n",
    green(money(res.value())), se, res.scenarios(), STEPS);
CODE

banner "4 · The whole CVA risk vector, from ONE reverse sweep"
say "The same recorded valuation, differentiated once. IR delta and vega,"
say "counterparty CS01 in three tenor buckets, FX delta, recovery — all of it"
say "falls out of the single sweep that produced the number above."

run <<'CODE'
var g = res.gradient();
double lgd = ns.counterparty().lossGivenDefault();
void row(String k, double v) { System.out.printf(Locale.ROOT, "   %-22s %s%n", k, dol(v)); }
row("IR delta  (1bp)", (g.r0() + g.hwLevel()) * 1e-4);
row("IR vega", g.hwSigma() * mkt.hwSigma());
row("CS01 short (1bp)", g.hazardShort() / lgd * 1e-4);
row("CS01 mid   (1bp)", g.hazardMid() / lgd * 1e-4);
row("CS01 long  (1bp)", g.hazardLong() / lgd * 1e-4);
row("FX delta  (1%)", g.fxSpot() * mkt.fxSpot() * 0.01);
CODE

run <<'CODE'
System.out.printf(Locale.ROOT, "   sweep %s s  (value + full gradient)   build %s s%n",
    yellow(String.format("%.3f", res.sweepSeconds())),
    String.format("%.3f", res.buildSeconds()));
CODE

banner "5 · THE HEAVY ALTERNATIVE: bump every risk factor, re-simulate"
note "the letter-compliant SA-CVA sensitivity: one full exposure re-run per shock"

run <<'CODE'
var bump = SaCvaSensitivities.bumpAndRevalue(sim, mkt, N, SEED, keys);
System.out.printf(Locale.ROOT, "   %d full re-simulations in %s s   vs one %s s sweep%n",
    bump.revaluations(), yellow(String.format("%.2f", bump.seconds())),
    yellow(String.format("%.2f", res.sweepSeconds())));
CODE

banner "6 · Reconcile: one sweep vs every re-simulation"
say "Same seed, same paths, same tape — the two agree to the bump's own"
say "second-order error."

run <<'CODE'
var adj = SaCvaSensitivities.adjoint(res, keys);
for (var e : bump.sensitivities().asMap().entrySet()) {
  var k = e.getKey();
  String tag = k.riskClass() + " " + k.measure() + " b" + k.bucket()
      + (k.tenor() > 0 ? " @" + k.tenor() + "y" : "");
  System.out.printf(Locale.ROOT, "   %-28s bump %+11.2f   sweep %+11.2f%n",
      tag, e.getValue(), adj.get(k));
}
CODE

banner "7 · Weight, correlate, LOW / MEDIUM / HIGH -> SA-CVA; then BA-CVA"
say "The netted sensitivities are risk-weighted and aggregated FRTB-style under"
say "three correlation scenarios. BA-CVA is a closed form on the exposure and"
say "the counterparty risk weight — cheap by comparison."

run <<'CODE'
var nsB = CvaShowcase.nettingSetB();
var bld = Cva.of(mkt).add(ns, keys).add(nsB, CvaShowcase.riskFactorsFor(nsB));
var cap = bld.hedge(CvaShowcase.hedgeOnA()).steps(STEPS).paths(N).seed(SEED).on(ENGINE).compute();
var sa = cap.saCva();
CODE

run <<'CODE'
System.out.printf(Locale.ROOT, "   SA-CVA   L %s   M %s   H %s   ->  %s  %s%n",
    money(sa.perScenario().get(CorrelationScenario.LOW)),
    money(sa.perScenario().get(CorrelationScenario.MEDIUM)),
    money(sa.perScenario().get(CorrelationScenario.HIGH)),
    sa.selected(), green(money(sa.total())));
System.out.printf(Locale.ROOT, "   BA-CVA   reduced %s   full %s   hedge benefit %s%n",
    money(cap.baCva().reduced()), money(cap.baCva().full()),
    money(cap.baCva().hedgeBenefit()));
CODE

banner "8 · The three PRA methods, and the binding charge"
say "Basel 3.1 gives UK banks no CVA internal model — only these three."

run <<'CODE'
var pra = PraCvaMethods.of(cap);
System.out.printf(Locale.ROOT, "   Alternative Approach   %s%n", money(pra.alternativeApproach()));
System.out.printf(Locale.ROOT, "   Basic Approach         %s%n", money(pra.basicApproach()));
System.out.printf(Locale.ROOT, "   Standardised (SA-CVA)  %s%n", money(pra.standardisedApproach()));
System.out.printf(Locale.ROOT, "   binding                %s%n", green(money(pra.bindingCharge())));
CODE

run <<'CODE'
System.out.printf(Locale.ROOT, "   adjoint work %.2f s across %d netting sets; the rest is arithmetic.%n",
    cap.sweepSeconds(), cap.perNettingSet().size());
CODE

finale "The exposure simulation is the only expensive stage." \
       "One sweep gave every CVA sensitivity; the bump route re-ran it 14 times." \
       "SA-CVA, BA-CVA reduced and full, and the three PRA methods, end to end."
