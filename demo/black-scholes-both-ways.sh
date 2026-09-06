#!/usr/bin/env bash
# The closed form and a hundred million simulated futures, meeting at one price.
#   ./demo/black-scholes-both-ways.sh [--fast] [--cpu]
DEMO_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_player.sh"

player_start "Black-Scholes, both ways" \
             "one closed form, a hundred million paths, the same number"

quiet <<'SETUP'
import com.nablatensor.quant.analytic.*;

long N = Long.getLong("nablatensor.demo.paths", GPU ? 100_000_000L : 5_000_000L);
long SEED = 42L;

// Act 1 — the analytic oracle, one number per line.
void one(String label, double v, String note) {
  System.out.printf(Locale.ROOT, "   %s%s   %s%n",
      silver(String.format("%-8s", label)),
      white(String.format("%16.5f", v)),
      grey(note));
}

// Act 2 — adjoint MC beside the closed form, with the gap.
void two(String label, double a, double b) {
  System.out.printf(Locale.ROOT, "   %s%s%s%s%n",
      silver(String.format("%-8s", label)),
      white(String.format("%16.5f", a)),
      plain(String.format("%16.5f", b)),
      grey(String.format("%14.2e", Math.abs(a - b))));
}

void rate(Nabla.TypedValuation<EquityMarket> p) {
  System.out.printf(Locale.ROOT, "   %s%n", grey(String.format(Locale.ROOT,
      "%,d paths, one run, on %s   (the per-run rate is measured just below)",
      p.scenarios(), mc.engine())));
}

// Median of the last `lastK` samples of `xs` — the settled per-run time once
// the one-off (compile, GPU clock spin-up) has been paid on the first call.
double settled(double[] xs, int lastK) {
  double[] tail = java.util.Arrays.copyOfRange(xs, xs.length - lastK, xs.length);
  java.util.Arrays.sort(tail);
  return tail[tail.length / 2];
}

// Act 2 — the headline: which device, and the settled throughput of the
// value + 5-Greeks reverse sweep, phrased for the eye.
String deviceLabel() {
  if (!GPU) return ENGINE + " (no GPU on this machine)";
  if (!ENGINE_DESC.contains(" \u00b7 ")) return ENGINE;
  return ENGINE_DESC.split(" \u00b7 ")[1].replaceAll("\\s*\\(.*\\)", "").trim();
}
String headline(double mPathsPerSec) {
  String dev = deviceLabel();
  if (mPathsPerSec >= 1000.0) {
    return String.format(Locale.ROOT,
        "   %.1f billion Black-Scholes paths/s  —  with every Greek  —  on %s",
        mPathsPerSec / 1000.0, dev);
  }
  return String.format(Locale.ROOT,
      "   %.0f million Black-Scholes paths/s  —  with every Greek  —  on %s",
      mPathsPerSec, dev);
}

// Act 2 — the five Greeks by central difference off a price-only kernel.
double cdiff(MonteCarlo<EquityMarket> po, long n,
             UnaryOperator<EquityMarket> up, UnaryOperator<EquityMarket> dn, double h) {
  double hi = po.run(up.apply(market), n, SEED).price();
  double lo = po.run(dn.apply(market), n, SEED).price();
  return (hi - lo) / h;
}

// The whole bump-and-revalue set: base + 2 replays per input = 11 passes.
// Returns the bump delta, for the three-way reconciliation line.
double bumpAll(MonteCarlo<EquityMarket> po, long n) {
  double s = market.spot(), k = market.strike(), v = market.vol();
  po.run(n, SEED);
  double d = cdiff(po,n, m->m.withSpot(s*1.005),  m->m.withSpot(s*0.995),  s*.01);
  cdiff(po,n, m->m.withVol(v+1e-4),    m->m.withVol(v-1e-4),    2e-4);
  cdiff(po,n, m->m.withRate(.0301),    m->m.withRate(.0299),    2e-4);
  cdiff(po,n, m->m.withStrike(k*1.005),m->m.withStrike(k*0.995),k*.01);
  cdiff(po,n, m->m.withMaturity(1.001),m->m.withMaturity(0.999),2e-3);
  return d;
}

// Act 3 — the standard error of an N-path European-call estimate, from the
// closed form: SE = e^{-rT} sqrt(Var[(S_T-K)^+]) / sqrt(N), with the second
// moment E[((S_T-K)^+)^2] in closed form for the lognormal S_T.
double callStdErr(EquityMarket m, double bsPrice, long n) {
  double s = m.spot(), k = m.strike(), t = m.maturity(), r = m.rate(), v = m.vol();
  double disc = Math.exp(-r * t), srt = v * Math.sqrt(t);
  double d1 = (Math.log(s / k) + (r + 0.5 * v * v) * t) / srt, d2 = d1 - srt;
  double e2 = s * s * Math.exp((2 * r + v * v) * t) * BlackScholes.N(d1 + srt)
            - 2 * k * s * Math.exp(r * t) * BlackScholes.N(d1)
            + k * k * BlackScholes.N(d2);
  double undisc = Math.exp(r * t) * bsPrice;
  double var = disc * disc * (e2 - undisc * undisc);
  return Math.sqrt(var / n);
}
SETUP

# ── act 1 · the closed form ────────────────────────────────────────────────
banner "1 · The formula in the textbook"
say "An at-the-money one-year call: spot 100, strike 100, vol 20%, rate 3%."
say "Black-Scholes-Merton gives the price and every Greek as arithmetic —"
say "no simulation, no grid. This is the oracle everything else is checked on."

run <<'CODE'
var market = EquityMarket.atmOneYear();
CODE

run <<'CODE'
var bs = BlackScholes.of(OptionType.CALL, market);
CODE

run <<'CODE'
one("price", bs.price(), "per 100 notional");
one("delta", bs.delta(), "N(d1)");
one("vega",  bs.vega(),  "per unit vol, not per 1%");
one("rho",   bs.rho(),   "dV/dr");
one("dV/dK", bs.strikeSensitivity(), "strike sensitivity");
CODE

nap 0.7

# ── act 2 · the same option, simulated ────────────────────────────────────
banner "2 · The same option, one hundred million times"
say "Record the payoff once as plain Java. NablaTensor flattens it to a tape"
say "and compiles that tape to a kernel — the slow part, and it happens once."
say "Then one forward pass for the price, one reverse sweep for all five Greeks."

run <<'CODE'
var build = MonteCarlo.of(Products.europeanCall()).market(market).steps(1);
var mc = build.fp32().greeks().on(ENGINE).build();
CODE

run <<'CODE'
System.out.printf(Locale.ROOT, "   engine %s   %s tape nodes   %s%n",
    green(mc.engine()), white(mc.nodes()),
    grey(String.format(Locale.ROOT, "record %.0f ms, build %.0f ms — once",
        mc.recordSeconds() * 1e3, mc.buildSeconds() * 1e3)));
CODE

run <<'CODE'
var p = mc.run(N, SEED);
CODE

run <<'CODE'
var g = p.greeks();
System.out.println(cyan(String.format("   %-8s%16s%16s%14s",
    "", "adjoint MC", "closed form", "|diff|")));
two("price", p.price(), bs.price());
two("delta", g.spot(),  bs.delta());
two("vega",  g.vol(),   bs.vega());
two("rho",   g.rate(),  bs.rho());
two("dV/dK", g.strike(), bs.strikeSensitivity());
rate(p);
CODE

say "Price and five Greeks from one sweep. The gap to the closed form is"
say "Monte-Carlo error, not a modelling difference: this run is single"
say "precision, and its price lands about one standard error from the formula."
say "The next act tracks that gap as the path count grows."
nap 0.6

say "The Greeks the other way: a price-only kernel, re-priced for the base and"
say "twice per input for a central difference — eleven Monte-Carlo passes."
say "No warm-up games: we run the sweep two dozen times and report the first"
say "call and the settled per-run rate separately. The first pays a one-off —"
say "kernel dispatch, and on a GPU the clock spinning up from idle."

run <<'CODE'
var pob = MonteCarlo.of(Products.europeanCall()).market(market).steps(1);
var po = pob.fp32().priceOnly().on(ENGINE).build();
double bd = bumpAll(po, N);
CODE

run <<'CODE'
double[] sweep = new double[24];
for (int i=0;i<sweep.length;i++){ long t=now(); mc.run(N,SEED); sweep[i]=msSince(t); }
double[] bump = new double[6];
for (int i=0;i<bump.length;i++){ long t=now(); bumpAll(po,N); bump[i]=msSince(t); }
double firstMs = sweep[0];
double sweepMs = settled(sweep, 12);
double bumpMs  = settled(bump, 3);
double gMps = N / sweepMs / 1e3;
CODE

run <<'CODE'
System.out.printf(Locale.ROOT, "   bump delta %s   adjoint %s   closed form %s%n",
    plain(String.format("%.5f", bd)),
    plain(String.format("%.5f", p.greeks().spot())),
    plain(String.format("%.5f", bs.delta())));
System.out.printf(Locale.ROOT, "%s%n", yellow(headline(gMps)));
System.out.printf(Locale.ROOT, "%s%n", yellow(String.format(Locale.ROOT,
    "   first call %.0f ms (one-off), then %.1f ms/run   ·   %,d paths, value + 5 Greeks",
    firstMs, sweepMs, N)));
System.out.printf(Locale.ROOT, "%s%n", yellow(String.format(Locale.ROOT,
    "   1 sweep %.1f ms  vs  11 bump passes %.1f ms  =  %.1fx  (+2 passes per factor)",
    sweepMs, bumpMs, bumpMs / sweepMs)));
CODE

say "Same delta three ways. One reverse sweep carried all five Greeks for about"
say "the cost of a single price pass; bump-and-revalue is eleven passes and adds"
say "two more for every risk factor, while the sweep stays one pass."
nap 0.8

# ── act 3 · convergence ───────────────────────────────────────────────────
banner "3 · Watch the simulation converge onto the formula"
say "Same kernel, more paths — one run each. Beside the gap to the closed form,"
say "the standard error the closed form itself predicts for a run that size."
say "The gap shrinks as the paths rise, and error/SE hovers around 1 — the run"
say "is always about as far from the formula as the standard error predicts."

run <<'CODE'
long[] big   = {10_000L, 100_000L, 1_000_000L, 10_000_000L};
long[] small = {10_000L, 100_000L, 1_000_000L};
long[] Ns = GPU ? big : small;
System.out.println(cyan(String.format("   %16s%16s%14s%12s",
    "paths", "MC price", "|error|", "error/SE")));
for (long n : Ns) {
    var q = mc.run(n, SEED);
    double err = Math.abs(q.price() - bs.price());
    double se = callStdErr(market, bs.price(), n);
    System.out.printf(Locale.ROOT, "   %s%s%s%s%n",
        plain(String.format("%,16d", n)),
        white(String.format("%16.5f", q.price())),
        grey(String.format("%14.2e", err)),
        yellow(String.format("%12.2f", err / se)));
}
CODE

nap 0.9

# ── act 4 · implied volatility ────────────────────────────────────────────
banner "4 · Implied volatility — the derivative is already there"
say "A broker quotes this call at a price. What volatility does that imply?"
say "Newton's method needs dPrice/dvol at every step. That is vega, and vega"
say "is one component of the same reverse sweep — no finite differences."

run <<'CODE'
double quote = BlackScholes.of(OptionType.CALL, market.withVol(0.28)).price();
long IV_N = GPU ? 20_000_000L : 2_000_000L;
double sig = 0.15;
System.out.printf(Locale.ROOT, "   quote %s   start vol %s%n",
    white(String.format("%.5f", quote)), yellow(String.format("%.4f", sig)));
for (int it = 1; it <= 6; it++) {
    var v = mc.run(market.withVol(sig), IV_N, SEED);
    double f = v.price() - quote;
    double vg = v.greeks().vol();
    System.out.printf(Locale.ROOT, "   step %d   vol %s   price - quote %s%n",
        it, yellow(String.format("%.6f", sig)), grey(String.format("%+.2e", f)));
    if (Math.abs(f) < 1e-5) break;
    sig = sig - f / vg;
}
CODE

run <<'CODE'
System.out.printf(Locale.ROOT, "   Newton implied vol %s   %s%n",
    green(String.format("%.6f", sig)),
    grey("the last digits are Monte-Carlo error in the simulated price, not a bug"));
CODE

say "Now the same root with no derivative — bisection on price alone:"

run <<'CODE'
double blo = 0.05, bhi = 1.00;
int steps = 0;
while (bhi - blo > 1e-6) {
    double mid = 0.5 * (blo + bhi);
    double fm = mc.run(market.withVol(mid), IV_N, SEED).price() - quote;
    if (fm > 0) bhi = mid; else blo = mid;
    steps = steps + 1;
}
System.out.printf(Locale.ROOT, "   bisection %s steps to 1e-6, same root %s%n",
    white(steps), green(String.format("%.6f", 0.5 * (blo + bhi))));
CODE

say "A handful of Newton steps against twenty-odd, because the slope came out"
say "of the same reverse sweep as the price."
nap 0.8

# ── act 5 · two checks the closed form makes cheap ────────────────────────
banner "5 · Two sanity checks, straight off the closed form"
say "A continuous dividend yield q just shifts the cost of carry to r - q:"

run <<'CODE'
var g0 = GeneralizedBsm.of(OptionType.CALL,100.0,100.0,1.0,0.03,0.00,0.20).greeks();
var gq = GeneralizedBsm.of(OptionType.CALL,100.0,100.0,1.0,0.03,0.02,0.20).greeks();
System.out.printf(Locale.ROOT, "   q = 0%%   price %s        q = 2%%   price %s%n",
    white(String.format("%.5f", g0.price())),
    white(String.format("%.5f", gq.price())));
CODE

say "And put-call parity, the no-arbitrage identity from the options chapter:"

run <<'CODE'
var put = BlackScholes.of(OptionType.PUT, market);
double fwd = market.strike() * Math.exp(-market.rate() * market.maturity());
double parity = (bs.price() - put.price()) - (market.spot() - fwd);
System.out.printf(Locale.ROOT, "   C - P  -  (S - K e^{-rT})  =  %s   %s%n",
    white(String.format("%.2e", parity)), grey("(zero, to rounding)"));
CODE

run <<'CODE'
mc.close();
po.close();
CODE

finale "The closed form and a hundred million paths meet, to Monte-Carlo error." \
       "Vega for the implied-vol solve fell out of the sweep that gave delta." \
       "See also: demo/adjoint-vs-bump.sh"
