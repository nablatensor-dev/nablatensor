#!/usr/bin/env bash
# One reverse sweep vs eleven revaluations: the same Greeks, a fraction of the work.
#   ./demo/adjoint-vs-bump.sh [--fast] [--cpu]
DEMO_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_player.sh"

player_start "One reverse sweep vs eleven revaluations" \
             "value + five Greeks, two ways, same numbers"

quiet <<'SETUP'
long N = Long.getLong("nablatensor.demo.paths", 8_000_000L);
long SEED = 42L;
var market = EquityMarket.atmOneYear();
void greekRow(String how, double pr, double dl, double vg, double rh,
              double dk, double dt, double ms) {
  System.out.printf(Locale.ROOT,
      "  %s  price %s  delta %s  vega %s  rho %s  dV/dK %s  dV/dT %s  %s ms%n",
      cyan(String.format("%-8s", how)),
      white(String.format("%.5f", pr)),  white(String.format("%.5f", dl)),
      white(String.format("%.4f", vg)),  white(String.format("%.4f", rh)),
      white(String.format("%.5f", dk)),  white(String.format("%.4f", dt)),
      yellow(String.format("%.0f", ms)));
}
// central difference of a price-only kernel along one market field
double bump(MonteCarlo<EquityMarket> po, UnaryOperator<EquityMarket> up,
            UnaryOperator<EquityMarket> down, double denom) {
  double hi = po.run(up.apply(market), N, SEED).price();
  double lo = po.run(down.apply(market), N, SEED).price();
  return (hi - lo) / denom;
}
SETUP

banner "1 · The ask"
say "A desk wants an Asian call's value and its five first-order Greeks:"
say "delta, vega, rho, strike sensitivity, time decay. 252 fixings, eight"
say "million paths. Two ways to get there."
nap 0.6

banner "2 · Way one — bump and revalue"
say "Build a price-only kernel. Re-price it once for the base, then twice per"
say "input for a central difference: 1 + 2x5 = eleven replays."

run <<'CODE'
var poB = MonteCarlo.of(Products.asianCall()).market(market).steps(252);
var po = poB.fp32().priceOnly().on(ENGINE).build();
po.run(N, SEED);
CODE

run <<'CODE'
long t0 = now();
double base = po.run(market, N, SEED).price();
double S = market.spot(), K = market.strike();
double dS = bump(po, m -> m.withSpot(S*1.01),  m -> m.withSpot(S*0.99),  2*S*0.01);
double dV = bump(po, m -> m.withVol(m.vol()+1e-4), m -> m.withVol(m.vol()-1e-4), 2e-4);
double dR = bump(po, m -> m.withRate(m.rate()+1e-4), m -> m.withRate(m.rate()-1e-4), 2e-4);
double dK = bump(po, m -> m.withStrike(K*1.01), m -> m.withStrike(K*0.99), 2*K*0.01);
double dT = bump(po, m -> m.withMaturity(m.maturity()+1e-3), m -> m.withMaturity(m.maturity()-1e-3), 2e-3);
double bumpMs = msSince(t0);
greekRow("bump", base, dS, dV, dR, dK, dT, bumpMs);
CODE

say "Every one of those replays is a full Monte-Carlo pass. And the step size"
say "is a guess: too big biases the Greek, too small drowns it in path noise."
nap 0.7

banner "3 · Way two — adjoint"
say "Build the same tape with .greeks(). One forward pass, one reverse sweep,"
say "and the gradient comes out market-shaped: one number per field."

run <<'CODE'
var adjB = MonteCarlo.of(Products.asianCall()).market(market).steps(252);
var adj = adjB.fp32().greeks().on(ENGINE).build();
adj.run(N, SEED);
CODE

run <<'CODE'
long t1 = now();
var a = adj.run(N, SEED);
double adjMs = msSince(t1);
var g = a.greeks();
greekRow("adjoint", a.price(), g.spot(), g.vol(), g.rate(),
    g.strike(), g.maturity(), adjMs);
CODE

banner "4 · Same answer, and it does not grow"

run <<'CODE'
System.out.printf(Locale.ROOT, "   speedup here: %s   (eleven replays vs one)%n",
    green(String.format("%.1fx", bumpMs / adjMs)));
System.out.printf(Locale.ROOT, "   %s%n", grey(
    "adjoint replays = 1, always. bump replays = 1 + 2*inputs."));
System.out.printf(Locale.ROOT, "   %s%n", grey(
    "each new Greek adds two bump replays and nothing to the sweep."));
CODE

run <<'CODE'
po.close();
adj.close();
CODE

say "The Greeks agree to Monte-Carlo noise. The reverse sweep costs about one"
say "extra forward pass no matter how many inputs there are — so the more risk"
say "factors the position has, the wider this gap gets."

finale "value + every Greek, from one adjoint sweep" \
       "no step size to choose, no revaluation count to watch grow" \
       "See also: demo/greeks-on-gpu.sh"
