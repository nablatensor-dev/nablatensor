#!/usr/bin/env bash
# ISDA SIMM in full: the whole initial-margin model, written out stage by stage —
# the CRIF sensitivity vector, the concentration thresholds, the nested delta /
# vega / curvature aggregation, the product-class roll-up, and the annual
# backtest. The expensive stage — generating the full sensitivity set — is marked
# in an orange comment.
#   ./demo/isda-simm-full.sh [--fast] [--cpu]
# Runs on the fastest adjoint backend available (CUDA > Vulkan > ROCm > cpu-jit);
# override with NABLATENSOR_DEMO_ENGINE=cuda|vulkan|rocm|cpu-jit. The heavy stage
# sweeps a 24-trade netting set at 1.3M paths x 252 steps per trade on a GPU
# backend (6 trades x 400k on cpu-jit); size it with -Dnablatensor.demo.book=
# and -Dnablatensor.demo.paths=.
DEMO_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_player.sh"

player_start "ISDA SIMM in full, on the fastest backend" \
             "every stage of the initial-margin model, in code"

note "fastest adjoint backend here: $(capture <<'PROBE'
System.out.print(ENGINE + "  (" + ENGINE_DESC + ")");
PROBE
)"

quiet <<'SETUP'
import com.nablatensor.risk.*;
import com.nablatensor.examples.SimmShowcase;
import com.nablatensor.examples.SimmShowcase.ProductClass;
int BOOK = Integer.getInteger("nablatensor.demo.book", GPU ? 24 : 6);
long N = Long.getLong("nablatensor.demo.paths", GPU ? 1_300_000L : 400_000L);
long SEED = 42L;
int STEPS = 252;
int BACKTEST_DAYS = Integer.getInteger("nablatensor.demo.backtest", 750);
String orange(Object s) { return color("38;5;208", s); }
String money(double v) { return String.format(Locale.ROOT, "$%,.3fm", v); }
String num(double v) { return String.format(Locale.ROOT, "%.3f", v); }
void row(String k, double v) {
  System.out.printf(Locale.ROOT, "   %-26s %s%n", k, num(v));
}
SETUP

banner "1 · The non-cleared book, the classes, the rulebook"
say "One CRIF file for a portfolio of non-cleared derivatives with four"
say "counterparties. SIMM has four product classes (RatesFX, Credit, Equity,"
say "Commodity) built from six risk classes, each with delta, vega and"
say "curvature. One versioned parameter set."

run <<'CODE'
var params = SimmShowcase.parameters();
var book = SimmShowcase.book();
for (var t : book)
  System.out.printf(Locale.ROOT, "   %-20s %-13s %-10s side %+2.0f  $%.0fm%n",
      t.id(), t.counterparty(), t.product(), t.side(), t.notional() / 1e6);
CODE

run <<'CODE'
for (var pc : ProductClass.values())
  System.out.printf(Locale.ROOT, "   %-10s <- %s%n", pc, pc.riskClasses());
System.out.printf(Locale.ROOT, "   %s   %d buckets across %d risk classes%n",
    params.version(), params.bucketCount(), SimmShowcase.SIMM_CLASSES.size());
CODE

banner "2 · The SIMM parameter tables"
say "Every bucket carries a delta risk weight, a vega risk weight, a"
say "within-bucket correlation, an across-bucket correlation, and delta / vega"
say "concentration thresholds. Demo values — not the ISDA-published calibration."

run <<'CODE'
for (var rc : SimmShowcase.SIMM_CLASSES) {
  var tbl = params.tables().get(rc);
  var p = tbl.values().iterator().next();
  System.out.printf(Locale.ROOT, "   %-12s %2d buckets   dRW %.3f  vRW %.3f  rho %.2f  gamma %.2f%n",
      rc, tbl.size(), p.deltaRw(), p.vegaRw(), p.rho(), p.gamma());
}
CODE

banner "3 · Map the book to a CRIF sensitivity vector"
say "Each trade contributes delta, vega and curvature on the regulatory factor"
say "keys of its product's risk classes — the Common Risk Interchange Format"
say "row set every SIMM implementation consumes."

run <<'CODE'
for (var pc : ProductClass.values()) {
  int factors = 0;
  for (var rc : pc.riskClasses())
    factors += Math.min(2, params.tables().get(rc).size()) * 3;
  System.out.printf(Locale.ROOT, "   %-10s %d trades touch %d factor slots (delta+vega+curvature)%n",
      pc, book.stream().filter(t -> t.product().equals(pc.name())).count(), factors);
}
CODE

banner "4 · THE HEAVY PART: generate the full sensitivity set"
note "one compiled kernel replayed across the book, adjoint reverse sweep per trade"

run <<'CODE'
System.out.printf(Locale.ROOT, "   %d-trade netting set, %,d paths x %d steps per trade%n",
    BOOK, N, STEPS);
var heavy = SimmShowcase.runHeavySensitivities(ENGINE, true, BOOK, N, STEPS, SEED);
var crif = SimmShowcase.buildSensitivities(book, params, heavy);
System.out.printf(Locale.ROOT, "   one adjoint sweep per trade   %s s   (%,d path valuations)%n",
    yellow(num(heavy.sweepSeconds())), heavy.pathValuations());
System.out.printf(Locale.ROOT, "   -> CRIF: %d risk-factor sensitivities across %d SIMM classes%n",
    crif.asMap().size(), SimmShowcase.SIMM_CLASSES.size());
CODE

run <<'CODE'
System.out.println(orange("   // cost shape: trades x N_paths x N_steps x per-trade pricing"));
System.out.println(orange("   // SIMM needs delta+vega+curvature to EVERY risk factor, regenerated"));
System.out.println(orange("   // daily for margin calls; by bump that is one book revaluation per"));
System.out.println(orange("   // factor per day per counterparty. The adjoint sweep gets the lot at once."));
CODE

banner "5 · The prescribed-bump alternative, timed"
say "The letter-compliant sensitivity: shock each risk factor, revalue every"
say "trade in the book, difference. One full revaluation per factor per trade;"
say "the sweep produced the whole vector in a single pass."

run <<'CODE'
System.out.printf(Locale.ROOT, "   %d bump revaluations     %s s%n",
    heavy.revaluations(), yellow(num(heavy.bumpSeconds())));
System.out.printf(Locale.ROOT, "   one adjoint sweep        %s s   for the entire vector%n",
    yellow(num(heavy.sweepSeconds())));
CODE

run <<'CODE'
System.out.println(orange("   // this book touches ~10 risk factors; a real one carries 100s -"));
System.out.println(orange("   // the bump cost scales with factor count, the sweep does not."));
CODE

banner "6 · Reconcile: the sweep's Greeks vs the bumped ones"
say "Same markets, same paths, same tape. The book delta and vega from the one"
say "reverse sweep match the central-difference bumps to their own truncation"
say "error."

run <<'CODE'
System.out.printf(Locale.ROOT, "   %-22s sweep %+11.4f   bump %+11.4f%n",
    "book delta", heavy.delta(), heavy.bumpDelta());
System.out.printf(Locale.ROOT, "   %-22s sweep %+11.4f   bump %+11.4f%n",
    "book vega (per 1%)", heavy.vega(), heavy.bumpVega());
row("book rho (per 1%)", heavy.irDelta());
row("book curvature", heavy.curvature());
CODE

banner "7 · Concentration: CR_b = max(1, sqrt(|S_b| / T_b))"
say "SIMM scales every weighted sensitivity in a bucket by a concentration"
say "factor, and multiplies the within-bucket cross terms by f_kl ="
say "min(CR_k,CR_l) / max(CR_k,CR_l). NestedAggregation carries both hooks."
say "A book this small stays under the thresholds, so CR_b = 1 here; a"
say "concentrated position is where the factor would bite."

run <<'CODE'
var girrDelta = crif.ofClass(RiskClass.GIRR).ofMeasure(RiskMeasure.DELTA);
var sums = new java.util.LinkedHashMap<String, Double>();
for (var e : girrDelta.asMap().entrySet())
  sums.merge(e.getKey().bucket(), e.getValue(), Double::sum);
double tG = params.tables().get(RiskClass.GIRR).values().iterator().next().deltaThreshold();
for (var e : sums.entrySet())
  System.out.printf(Locale.ROOT, "   bucket %-4s  S_b %+9.3f  T_b %.0f  CR_b %.3f%n",
      e.getKey(), e.getValue(), tG, Math.max(1.0, Math.sqrt(Math.abs(e.getValue()) / tG)));
CODE

banner "8 · Delta margin per risk class"
say "WS_k = RW_k * s_k * CR_k, aggregated inside each bucket with rho, then"
say "across buckets with gamma. Shown inline for GIRR, then every class."

run <<'CODE'
var tGirr = params.tables().get(RiskClass.GIRR);
double gammaGirr = tGirr.values().iterator().next().gamma();
var aggGirr = NestedAggregation.delta(
    f -> tGirr.get(f.bucket()).deltaRw(),
    (k, l) -> k.equals(l) ? 1.0 : tGirr.get(k.bucket()).rho(),
    (b, c) -> b.equals(c) ? 1.0 : gammaGirr);
var kGirr = aggGirr.aggregate(girrDelta);
System.out.printf(Locale.ROOT, "   GIRR delta margin  K = %s   (%d buckets)%n",
    num(kGirr.total()), kGirr.kb().size());
CODE

run <<'CODE'
var res = SimmShowcase.simm(params, crif);
for (var rc : SimmShowcase.SIMM_CLASSES)
  System.out.printf(Locale.ROOT, "   %-12s delta %8.3f%n", rc,
      res.byClass().get(rc).get(RiskMeasure.DELTA));
CODE

banner "9 · Vega and curvature margin per risk class"
say "Same nesting, vega risk weights and vega thresholds for vega; the"
say "curvature variant squares the correlations and drops diversification"
say "between same-sign shocks."

run <<'CODE'
for (var rc : SimmShowcase.SIMM_CLASSES)
  System.out.printf(Locale.ROOT, "   %-12s vega %8.3f   curvature %8.3f%n", rc,
      res.byClass().get(rc).get(RiskMeasure.VEGA),
      res.byClass().get(rc).get(RiskMeasure.CURVATURE));
CODE

banner "10 · IM per product class"
say "Within a product class the risk-class margins are combined with the SIMM"
say "risk-class correlation psi: IM = sqrt( sum K_r^2 + sum psi_rs K_r K_s )."
say "RatesFX pairs IR with FX (psi 0.28); Credit pairs the two credit classes."

run <<'CODE'
for (var pc : ProductClass.values()) {
  var im = res.byProduct().get(pc);
  System.out.printf(Locale.ROOT, "   %-10s delta %7.3f  vega %7.3f  curv %7.3f  ->  %s%n",
      pc, im.get(RiskMeasure.DELTA), im.get(RiskMeasure.VEGA),
      im.get(RiskMeasure.CURVATURE), money(res.productTotal().get(pc)));
}
CODE

banner "11 · Total SIMM, and the annual backtest"
say "SIMM is the simple sum of the four product-class margins. The ISDA backtest"
say "then recomputes IM and clean P&L over years of daily history for every"
say "counterparty — the second place the compute goes."

run <<'CODE'
System.out.printf(Locale.ROOT, "   total SIMM initial margin   %s%n", green(money(res.total())));
var bt = SimmShowcase.backtest(params, crif, BACKTEST_DAYS, SEED);
System.out.printf(Locale.ROOT, "   backtest  %d snapshots in %s s   %d P&L breaches%n",
    bt.days(), num(bt.seconds()), bt.breaches());
CODE

run <<'CODE'
double engineMin = BACKTEST_DAYS * heavy.sweepSeconds() / 60.0;
System.out.println(orange("   // the loop above only re-aggregates; a real backtest regenerates the"));
System.out.println(orange(String.format(Locale.ROOT,
    "   // sensitivities per historical date - ~%.0f min of sweeps here, far more by bump.", engineMin)));
CODE

banner "12 · Where the time went"
say "One adjoint sweep produced the whole sensitivity vector that every margin"
say "above is built from. The bump route reruns the book valuation once per"
say "factor; the nested aggregation and the product roll-up are arithmetic."

run <<'CODE'
System.out.printf(Locale.ROOT, "   sweep %s s   vs   %d bump revaluations %s s%n",
    num(heavy.sweepSeconds()), heavy.revaluations(), num(heavy.bumpSeconds()));
System.out.println(grey("   Calculators, not sign-off - model validation and margin disputes stay with you."));
CODE

finale "Generating the full sensitivity set was the one expensive stage." \
       "One sweep gave every SIMM delta, vega and curvature; the bump route re-ran the book per factor." \
       "CRIF, concentration, nested aggregation, product roll-up and the backtest, end to end."
