#!/usr/bin/env bash
# Adjoint AD, compiled to the GPU: a barrier note priced with every Greek.
#   ./demo/greeks-on-gpu.sh [--fast] [--cpu]
DEMO_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_player.sh"

player_start "Adjoint AD, compiled to the GPU" \
             "record a valuation once, replay it twenty million times"

quiet <<'SETUP'
void row(String label, double v, String note) {
  System.out.printf(Locale.ROOT, "   %s %16s   %s%n",
      silver(String.format("%-9s", label)),
      white(String.format(Locale.ROOT, "%,.4f", v)),
      grey(note));
}
void rate(Nabla.TypedValuation<EquityMarket> p) {
  System.out.printf("   %s%n", grey(String.format(Locale.ROOT,
      "%,d paths in %.2f s  =  %.2e paths/s  on %s",
      p.scenarios(), p.seconds(), p.scenariosPerSecond(), mc.engine())));
}
SETUP

# ── act 1 · the position ────────────────────────────────────────────────────
banner "1 · The position"
say "A desk holds a down-and-in put: strike 100, knock-in at 70. Today it is"
say "worthless paper — but if the stock ever trades through 70 it springs to"
say "life as a put and the desk is long a lot of downside."
say ""
say "What is it worth, and what hedges it?"
nap 0.8

# ── act 2 · the market ──────────────────────────────────────────────────────
banner "2 · The market is a record. Every field is a risk factor."

run <<'CODE'
var market = new EquityMarket(100.0, 100.0, 0.28, 0.03, 1.0);
CODE

say "Five doubles: spot, strike, vol, rate, maturity. No framework, no"
say "annotations, no risk factor named with a string. EquityMarket::vol is"
say "checked by the compiler and impossible to misspell."

# ── act 3 · the payoff ─────────────────────────────────────────────────────
banner "3 · The payoff is a lambda. Plain Java. No AD, no GPU."

run <<'CODE'
var note = ExoticProducts.barrier(
    OptionType.PUT, ExoticProducts.Barrier.DOWN_IN, 70.0, 1.0);
CODE

say "A down-and-in put, monitored every step with a smoothed indicator so the"
say "whole payoff stays differentiable. The barrier at 70 is just a number in"
say "the lambda — but its sensitivity still falls out of the same sweep."

# ── act 4 · record + compile ───────────────────────────────────────────────
banner "4 · Record once. Compile once. Then never again."
note "tape -> SPIR-V compute shader -> Vulkan ... the slow part, and it happens once"

run <<'CODE'
var build = MonteCarlo.of(note).market(market).steps(252);
var mc = build.fp32().greeks().on(ENGINE).build();
CODE

run <<'CODE'
System.out.printf("   engine %s   %s tape nodes%n",
    green(mc.engine()), white(mc.nodes()));
CODE

# ── act 5 · the money ──────────────────────────────────────────────────────
banner "5 · Price it. Twenty million futures. Every Greek at once."

run <<'CODE'
var p = mc.run(20_000_000L, 42L);
CODE

run <<'CODE'
var g = p.greeks();
row("price",  p.price(),      "per 100 notional");
row("delta",  g.spot(),       "<-- shares to hedge with. THIS is the number.");
row("vega",   g.vol(),        "long vol, as a down-and-in put is");
row("rho",    g.rate(),       "rate sensitivity");
row("dV/dK",  g.strike(),     "strike sensitivity");
row("dV/dT",  g.maturity(),   "time decay");
rate(p);
CODE

say "Five sensitivities, one simulation. Bump-and-revalue needs a full re-run"
say "per input, and the count only ever grows."
nap 0.9

# ── act 6 · the ladder ────────────────────────────────────────────────────
banner "6 · Move the market. Same kernel, no rebuild — it is an argument."

run <<'CODE'
System.out.println(cyan("    spot        price        delta         vega"));
for (double s = 80; s <= 120; s += 10) {
    var q = mc.run(market.withSpot(s), 5_000_000L, 42L);
    var qg = q.greeks();
    System.out.printf(Locale.ROOT, "  %s %s %s %s%n",
        yellow(String.format("%6.0f", s)),
        white(String.format("%12.4f", q.price())),
        plain(String.format("%12.4f", qg.spot())),
        plain(String.format("%12.3f", qg.vol())));
}
CODE

say "Delta runs from -0.88 near the barrier to -0.12 far above it: close to 70"
say "the option is alive and tracks the put; up at 120 the knock-in rarely"
say "triggers. That curvature is the barrier risk, and it is exactly where"
say "bump-and-revalue returns noise instead of a number."
nap 1.0

# ── act 7 · the crash ────────────────────────────────────────────────────
banner "7 · Now the market gaps down. Spot 72, vol 55%."

run <<'CODE'
var crashed = market.withSpot(72.0).withVol(0.55);
var c = mc.run(crashed, 20_000_000L, 42L);
var cg = c.greeks();
row("price", c.price(),  "worth ~6 a note this morning; knock-in now near-certain");
row("delta", cg.spot(),  "<-- -0.38 -> -0.64 in one gap. Sell more stock.");
row("vega",  cg.vol(),   "");
CODE

run <<'CODE'
mc.close();
CODE

finale "The quant wrote a payoff.  The desk got a hedge that moves with it." \
       "Neither of them typed the word Vulkan." \
       "See also: demo/adjoint-vs-bump.sh"
