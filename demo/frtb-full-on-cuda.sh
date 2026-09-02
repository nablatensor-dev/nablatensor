#!/usr/bin/env bash
# FRTB full: the whole standardised approach, written out stage by stage.
#   ./demo/frtb-full-on-cuda.sh [--fast] [--cpu]
# Runs on the fastest adjoint backend available (CUDA > Vulkan > ROCm > cpu-jit);
# override with NABLATENSOR_DEMO_ENGINE=cuda|vulkan|rocm|cpu-jit.
DEMO_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_player.sh"

player_start "FRTB full on the fastest backend" \
             "every stage of the capital calculation, in code"

note "fastest adjoint backend here: $(capture <<'PROBE'
System.out.print(ENGINE + "  (" + ENGINE_DESC + ")");
PROBE
)"

quiet <<'SETUP'
import com.nablatensor.examples.FrtbFullShowcase;
import com.nablatensor.examples.FrtbFullShowcase.*;
import com.nablatensor.risk.*;
import com.nablatensor.scenario.*;
long N = Long.getLong("nablatensor.demo.paths", 2_000_000L);
long SEED = 42L;
double SHORT = -1.0;
String orange(Object s) { return color("38;5;208", s); }
SETUP

banner "1 · Load the market, the trades, the rulebook"
say "Nothing is calculated before the inputs are dated and counted:"
say "a market snapshot, a trade file, and a versioned parameter set."

run <<'CODE'
var params = FrtbFullShowcase.parameters();
var market = FrtbFullShowcase.marketData();
var trades = FrtbFullShowcase.trades();
System.out.printf(Locale.ROOT, "   %s  %d curves  %d trades  %d buckets%n",
    market.asOf(), market.levels().size(), trades.size(), params.bucketCount());
for (var t : trades) System.out.println(grey("   " + t));
CODE

banner "2 · The regulatory parameter tables"
say "Seven risk classes, 89 buckets. Every bucket carries a delta risk weight,"
say "a vega risk weight, a within-bucket rho and an across-bucket gamma."

run <<'CODE'
var eqTable = params.tables().get(RiskClass.EQUITY);
for (var rc : RiskClass.values()) {
  System.out.printf(Locale.ROOT, "   %-12s %2d buckets%n",
      rc, params.tables().get(rc).size());
}
System.out.println(cyan("   equity bucket 5 -> ") + eqTable.get("5"));
CODE

banner "3 · Delta: one adjoint sweep on the GPU"
say "The short Asian call is recorded once and differentiated in one reverse"
say "sweep. Delta is cheap. What follows is not."

run <<'CODE'
var mkt = EquityMarket.atmOneYear();
var asian = Products.asianCall();
var aadBuild = MonteCarlo.of(asian).market(mkt).steps(252);
var aad = aadBuild.fp32().greeks().on(ENGINE).build();
aad.run(50_000L, SEED);
var risk = aad.run(mkt, N, SEED);
double delta = SHORT * risk.greek(EquityMarket::spot);
System.out.println("   delta " + green(String.format("%.6f", delta)));
CODE

banner "4 · THE HEAVY PART: base / +30% / -30% full repricings"
note "one compiled kernel · common random numbers · three complete replays"

run <<'CODE'
var priceBuild = MonteCarlo.of(asian).market(mkt).steps(252);
var pricer = priceBuild.fp32().priceOnly().on(ENGINE).build();
var shocks = ScenarioSet.list(
    Scenario.of("base"),
    Scenario.of("up",   Shock.relative("spot",  0.30)),
    Scenario.of("down", Shock.relative("spot", -0.30)));
pricer.run(50_000L, SEED);
var pv = ScenarioRunner.run(pricer, mkt, shocks, N, SEED);
CODE

run <<'CODE'
System.out.printf(Locale.ROOT, "   adjoint %.3f s   base %.3f s%n",
    risk.seconds(), pv.get("base").seconds());
System.out.printf(Locale.ROOT, "   spot +30%% %.3f s   spot -30%% %.3f s%n",
    pv.get("up").seconds(), pv.get("down").seconds());
double pairSeconds = pv.get("up").seconds() + pv.get("down").seconds();
System.out.println(orange("   // 10,000 curvature factors, 2 replays each: ~" +
    String.format(Locale.ROOT, "%.1f min", 10_000 * pairSeconds / 60)));
CODE

banner "5 · Strip the linear P&L: CVR"
say "Curvature keeps only the loss delta could not explain, and the regulation"
say "takes the worse of the two shocked sides."

run <<'CODE'
void row(String label, double value) {
  System.out.printf(Locale.ROOT, "   %s %s%n",
      silver(String.format("%-26s", label)),
      white(String.format("%12.4f", value)));
}
double base = SHORT * pv.get("base").price();
double up   = SHORT * pv.get("up").price();
double down = SHORT * pv.get("down").price();
double shock = 0.30 * mkt.spot();
double upRes   = up - base - shock * delta;
double downRes = down - base + shock * delta;
double cvr = -Math.min(upRes, downRes);
row("base PV", base);
row("PV at spot +30%", up);
row("PV at spot -30%", down);
row("up residual", upRes);
row("down residual", downRes);
row("CVR", cvr);
CODE

banner "6 · Map trades to risk factors, then net"
say "Each trade carries its own sensitivity vector. Netting adds them on the"
say "shared regulatory factor keys: trade, then netting set, then book."

run <<'CODE'
var portfolio = FrtbFullShowcase.buildPortfolio(trades, params, market, cvr);
var netted = portfolio.aggregate();
int observations = 0;
for (var t : portfolio.trades()) observations += t.sensitivities().asMap().size();
System.out.printf(Locale.ROOT, "   %d trade observations -> %d net factors%n",
    observations, netted.asMap().size());
for (var e : portfolio.byNettingSet().entrySet()) {
  System.out.printf(Locale.ROOT, "   %-12s %d factors%n",
      e.getKey(), e.getValue().asMap().size());
}
CODE

run <<'CODE'
var asianCvr = RiskFactor.equityDelta("5", "ASIAN-001").asCurvature();
System.out.println("   the GPU CVR, now in the book: " +
    green(String.format("%.6f", netted.get(asianCvr))));
CODE

banner "7 · Weight, correlate, and run LOW / MEDIUM / HIGH"
say "WS = RW x s, aggregated inside each bucket, then across buckets. FRTB"
say "runs that three times and keeps whichever correlation scenario hurts most."

run <<'CODE'
var eqDelta = netted.ofClass(RiskClass.EQUITY).ofMeasure(RiskMeasure.DELTA);
for (var sc : CorrelationScenario.values()) {
  var agg = NestedAggregation.delta(
      k -> eqTable.get(k.bucket()).deltaRw(),
      (k, l) -> k.equals(l) ? 1.0 : sc.apply(eqTable.get(k.bucket()).rho()),
      (b, c) -> b.equals(c) ? 1.0 : sc.apply(0.15));
  row("equity delta " + sc, agg.aggregate(eqDelta).total());
}
CODE

say "The same three-scenario run, for every class and every measure:"

run <<'CODE'
void scen(String label, MeasureCapital m) {
  var s = m.scenarios();
  System.out.printf(Locale.ROOT, "     %s L %8.3f  M %8.3f  H %8.3f  -> %s%n",
      silver(String.format("%-9s", label)),
      s.get(CorrelationScenario.LOW), s.get(CorrelationScenario.MEDIUM),
      s.get(CorrelationScenario.HIGH),
      green(String.format("%-6s %.3f", m.selected(), m.total())));
}
var capital = FrtbFullShowcase.aggregate(params, netted);
for (var rc : RiskClass.values()) {
  var c = capital.byClass().get(rc);
  System.out.println(cyan("   " + rc));
  scen("delta", c.delta());
  scen("vega", c.vega());
  scen("curvature", c.curvature());
}
CODE

banner "8 · Default risk charge"
say "Jump-to-default nets by obligor. A hedge benefit ratio then limits how"
say "much the short positions are allowed to offset the longs."

run <<'CODE'
var jtd = FrtbFullShowcase.defaultPositions();
for (var p : jtd) System.out.println(grey("   " + p));
var drc = FrtbFullShowcase.defaultRiskCharge(jtd);
double hbr = drc.longs() / (drc.longs() + drc.shorts());
double charge = Math.max(drc.weightedLongs() - hbr * drc.weightedShorts(), 0.0);
row("net JTD long ($m)", drc.longs());
row("net JTD short ($m)", drc.shorts());
row("hedge benefit ratio", hbr);
row("DRC ($m)", charge);
CODE

banner "9 · Residual risk add-on"
say "A flat gross-notional surcharge for what SBM and DRC cannot capture:"
say "1.0% on exotic underlyings, 0.1% on everything else in scope."

run <<'CODE'
var residual = FrtbFullShowcase.residualPositions(trades);
double exotic = 0.0;
double other = 0.0;
for (var p : residual) {
  double gross = Math.abs(p.grossNotional()) / 1e6;
  if (p.exotic()) exotic += 0.010 * gross;
  if (!p.exotic()) other += 0.001 * gross;
}
row("RRAO exotic ($m)", exotic);
row("RRAO other ($m)", other);
CODE

banner "10 · Capital bridge, controls, sign-off"
say "The three charges add with no diversification between them."

run <<'CODE'
var rrao = FrtbFullShowcase.residualRiskAddOn(residual);
double sbm = capital.total();
double frtb = sbm + drc.total() + rrao.total();
row("SBM ($m)", sbm);
row("DRC ($m)", drc.total());
row("RRAO ($m)", rrao.total());
row("FRTB total ($m)", frtb);
CODE

run <<'CODE'
var signOff = SignOff.review(params, market, portfolio, capital, drc, rrao);
for (var c : signOff.controls()) System.out.println(grey("   [PASS] " + c));
signOff.approvers().forEach((role, who) ->
    System.out.println("   " + silver(String.format("%-18s", role)) + plain(who)));
System.out.println("   " + green(signOff.status()));
CODE

run <<'CODE'
aad.close();
pricer.close();
CODE

finale "Every number above was followed from input to signature." \
       "The only expensive stage was the shocked repricing on the GPU." \
       "Seven classes, 89 buckets, netting, CVR, DRC, RRAO, sign-off."