#!/usr/bin/env bash
# One recording, every backend: the same tape, the same numbers, honestly timed.
#   ./demo/one-tape-every-backend.sh [--fast]
DEMO_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_player.sh"

player_start "One tape, every backend" \
             "discovered not configured · agrees with the oracle · timed without lying"

quiet <<'SETUP'
var MKT = EquityMarket.atmOneYear();
int STEPS = 252;
long N = Long.getLong("nablatensor.demo.paths", 2_000_000L);
long SEED = 42L;
double ORACLE_PX = Double.NaN, ORACLE_DL = Double.NaN;

void header() {
  System.out.printf(Locale.ROOT, "  %s%n", cyan(String.format(
      "%-10s %13s %13s   %9s %9s   %11s", "engine", "price", "delta",
      "d price", "d delta", "scen/s")));
}
void row(String eng, double px, double dl, double sps, boolean isOracle) {
  String dpx = isOracle ? "oracle" : String.format(Locale.ROOT, "%.1e", Math.abs(px - ORACLE_PX));
  String ddl = isOracle ? "" : String.format(Locale.ROOT, "%.1e", Math.abs(dl - ORACLE_DL));
  System.out.printf(Locale.ROOT, "  %s %s %s   %s %s   %s%n",
      cyan(String.format("%-10s", eng)),
      white(String.format("%13.6f", px)), white(String.format("%13.6f", dl)),
      grey(String.format("%9s", dpx)), grey(String.format("%9s", ddl)),
      yellow(String.format("%11.2e", sps)));
}
// warm once (first call pays codegen + device init), then time the second run
void replay(String eng, boolean fp32) {
  try {
    var b = MonteCarlo.of(Products.asianCall()).market(MKT).steps(STEPS);
    b = fp32 ? b.fp32() : b.fp64();
    var mc = b.greeks().on(eng).build();
    mc.run(N, SEED);
    var p = mc.run(N, SEED);
    row(eng, p.price(), p.greeks().spot(), p.scenariosPerSecond(), false);
    mc.close();
  } catch (RuntimeException | LinkageError e) {
    System.out.printf(Locale.ROOT, "  %-10s (skipped: %s)%n", eng, e.getMessage());
  }
}
SETUP

# ── act 1 ──────────────────────────────────────────────────────────────────
banner "1 · Backends are discovered, not configured"

run <<'CODE'
for (var e : AadEngines.discovered())
    System.out.printf(Locale.ROOT, "  %-12s %s  %s%n", e.name(),
        e.isAvailable() ? green("ok") : grey("--"), grey(e.describe()));
CODE

say "A ServiceLoader finds them. Every backend module is pure java.lang.foreign"
say "with runtime-compiled kernels, so it builds whether or not the GPU runtime"
say "is installed; isAvailable() decides at launch. No device, no Vector API —"
say "the row just says '--' and selection skips it."

# ── act 2 ──────────────────────────────────────────────────────────────────
banner "2 · Record the model once"
say "An arithmetic-average Asian call, 252 fixings, recorded against SDouble."
say "The tape below IS the model. Every backend replays these same nodes."

run <<'CODE'
var ob = MonteCarlo.of(Products.asianCall()).market(MKT).steps(STEPS);
var oracle = ob.fp64().greeks().on("cpu").build();
System.out.printf("   %s tape nodes%n", white(oracle.nodes()));
CODE

run <<'CODE'
oracle.run(N, SEED);
var op = oracle.run(N, SEED);
ORACLE_PX = op.price();
ORACLE_DL = op.greeks().spot();
header();
row("cpu", op.price(), op.greeks().spot(), op.scenariosPerSecond(), true);
oracle.close();
CODE

say "The scalar 'cpu' backend is the oracle: one IEEE-754 op at a time, in"
say "source order. Slow, and exactly right by definition."

# ── act 3 ──────────────────────────────────────────────────────────────────
banner "3 · The fp64 family reproduces it"

run <<'CODE'
for (String e : List.of("cpu-jit", "simd")) replay(e, false);
CODE

say "cpu-jit matches the oracle to rounding — the generated"
say "bytecode does the same ops in the same order. simd differs a little more,"
say "from reduction order, and every one of them runs several times faster."

# ── act 4 ──────────────────────────────────────────────────────────────────
banner "4 · fp32 on the GPU tracks it too"

run <<'CODE'
for (String e : List.of("vulkan", "rocm")) replay(e, true);
CODE

say "fp32 arithmetic and an on-device Philox stream, so these agree with the"
say "fp64 oracle to about five decimals on price and delta — comfortably"
say "inside Monte-Carlo noise at two million paths — for 10-20x the throughput."

# ── act 5 ──────────────────────────────────────────────────────────────────
banner "5 · Why those scen/s numbers are honest"

say "The replay helper does three things a quick benchmark usually skips:"
say ""
say "   warm up     the first call pays SPIR-V / bytecode codegen and GPU init"
say "   measure #2  time the run, not the one-off compile in front of it"
say "   same work   identical seed, scenario count and tape on every row"
say ""
say "So the speedup is one computation going faster — not a different, looser"
say "one. Cross-checked against the oracle every run; see docs/validation.md."

finale "Record once. Run it anywhere this box can. Same numbers." \
       "The backend is a string; the model never changed." \
       "See also: demo/greeks-on-gpu.sh"
