#!/usr/bin/env bash
# FRTB curvature: the expensive shocked repricings, then tiny post-processing.
#   ./demo/frtb-curvature-on-cuda.sh [--fast] [--cpu]
# Runs on the fastest adjoint backend available (CUDA > Vulkan > ROCm > cpu-jit);
# override with NABLATENSOR_DEMO_ENGINE=cuda|vulkan|rocm|cpu-jit.
DEMO_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_player.sh"

player_start "FRTB curvature on the fastest backend" \
             "two full shocked repricings are where the time goes"

note "fastest adjoint backend here: $(capture <<'PROBE'
System.out.print(ENGINE + "  (" + ENGINE_DESC + ")");
PROBE
)"

quiet <<'SETUP'
import com.nablatensor.scenario.*;
import com.nablatensor.risk.*;
long N = Long.getLong("nablatensor.demo.paths", 5_000_000L);
long SEED = 42L;
double shortPosition = -1.0;
String orange(Object s) { return color("38;5;208", s); }
String bankTime(double seconds) {
    return seconds < 3600
            ? String.format(Locale.ROOT, "%.1f min", seconds / 60)
            : String.format(Locale.ROOT, "%.1f h", seconds / 3600);
}
void timing(String label, Nabla.TypedValuation<EquityMarket> p) {
  System.out.printf(Locale.ROOT, "   %s %s   %s%n",
      silver(String.format("%-10s", label)),
      yellow(String.format("%6.2f s", p.seconds())),
      grey(String.format("%.2e paths/s", p.scenariosPerSecond())));
}
SETUP

banner "1 · The question"
say "A bank must measure non-linear loss under prescribed market shocks."
say "Delta is cheap with adjoint AD. Curvature still needs the portfolio"
say "priced again at spot up and spot down. That is the expensive part."

banner "2 · A path-dependent position"
say "Use a short Asian call: 252 daily fixings and five million paths."
say "The short position makes its convexity loss visible as a positive charge."

run <<'CODE'
var market = EquityMarket.atmOneYear();
var asian = Products.asianCall();
CODE

banner "3 · Delta: one adjoint run"

run <<'CODE'
var aadBuild = MonteCarlo.of(asian).market(market).steps(252);
var aad = aadBuild.fp32().greeks().on(ENGINE).build();
aad.run(50_000L, SEED);
var risk = aad.run(N, SEED);
double delta = shortPosition * risk.greek(EquityMarket::spot);
System.out.println("   delta " + green(String.format("%.6f", delta)));
timing("adjoint", risk);
CODE

say "One reverse sweep gives delta. Now comes the work adjoints cannot remove:"
say "the regulation asks for actual P&L after large up and down shocks."

banner "4 · THE HEAVY PART: three full GPU repricings"
note "base +30% -30%  ·  same compiled kernel  ·  same random paths"

run <<'CODE'
var priceBuild = MonteCarlo.of(asian).market(market).steps(252);
var pricer = priceBuild.fp32().priceOnly().on(ENGINE).build();
var shocks = ScenarioSet.list(
    Scenario.of("base"),
    Scenario.of("up",   Shock.relative("spot",  0.30)),
    Scenario.of("down", Shock.relative("spot", -0.30)));
pricer.run(50_000L, SEED);
var pv = ScenarioRunner.run(pricer, market, shocks, N, SEED);
CODE

run <<'CODE'
System.out.println(cyan("   replay       GPU time        throughput"));
timing("base", pv.get("base"));
timing("spot +30%", pv.get("up"));
timing("spot -30%", pv.get("down"));
double pairSeconds = pv.get("up").seconds() + pv.get("down").seconds();
double replaySeconds = pv.get("base").seconds() + pairSeconds;
System.out.println("\n   shocked repricing total " +
    yellow(String.format("%.2f s", replaySeconds)));
System.out.println(orange("   // Common bank book (~10,000 factors): about"));
System.out.println(orange("   // 10,000x longer here, ~" +
    bankTime(10_000 * pairSeconds) + " sequential; normally distributed."));
CODE

say "This loop dominates a bank run: two full repricings per risk factor."
say "The GPU accelerates each replay; banks distribute factors over a compute grid."

banner "5 · The FRTB logic afterwards is small"

run <<'CODE'
double base = shortPosition * pv.get("base").price();
double up   = shortPosition * pv.get("up").price();
double down = shortPosition * pv.get("down").price();
double shock = 0.30 * market.spot();
double cvr = -Math.min(up-base-shock*delta, down-base+shock*delta);
CODE

run <<'CODE'
var f = RiskFactor.equityDelta("5", "ASIAN").asCurvature();
var values = Sensitivities.builder().add(f, cvr).build();
var aggregation = NestedAggregation.curvature((a,b)->1, (a,b)->1);
double charge = aggregation.aggregate(values).total();
System.out.println("   CVR " + white(String.format("%.6f", cvr)));
System.out.println("   curvature charge " + green(String.format("%.6f", charge)));
CODE

say "A few subtractions, one minimum, then bucket aggregation. Compared with"
say "the Monte Carlo repricings above, this part is effectively free."

run <<'CODE'
aad.close();
pricer.close();
CODE

finale "Yes: curvature input generation is the heavy FRTB calculation." \
       "The cost is 2 shocked full repricings per factor. The GPU does that work." \
       "CVR and aggregation are the short epilogue, not the bottleneck."