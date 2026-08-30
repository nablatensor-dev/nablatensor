#!/usr/bin/env bash
# Fit a volatility smile with an adjoint gradient: one recorded objective, a reverse sweep per step.
#   ./demo/calibrate-a-smile.sh [--fast]
DEMO_ARGS=("$@")
source "$(dirname "${BASH_SOURCE[0]}")/_player.sh"

player_start "Calibrate a smile with an adjoint gradient" \
             "record the objective once, differentiate it every iteration"

quiet <<'SETUP'
double F = 100.0, T = 1.0, BETA = 1.0;
double[] STRIKES = {75.0, 85.0, 93.0, 99.0, 108.0, 121.0, 140.0};
double TRUE_ALPHA = 0.20, TRUE_RHO = -0.05, TRUE_NU = 1.0;
double[] target = new double[STRIKES.length];
for (int i = 0; i < STRIKES.length; i++)
  target[i] = SabrHagan.blackVol(TRUE_ALPHA, BETA, TRUE_RHO, TRUE_NU, F, STRIKES[i], T);

// A tiny fixed-width plot: strike on x, Black vol on y.
//   '*' = a market quote        '.' = the fitted SABR curve
void plotSmile(double[] k, double[] v, double[] fk, double[] fv) {
  int H = 11, Wd = 52;
  double x0 = k[0], x1 = k[k.length - 1];
  double y0 = Double.MAX_VALUE, y1 = -Double.MAX_VALUE;
  for (double y : v)  { y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  if (fv != null) for (double y : fv) { y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  double pad = (y1 - y0) * 0.15 + 1e-9; y0 -= pad; y1 += pad;
  char[][] g = new char[H][Wd];
  for (char[] row : g) Arrays.fill(row, ' ');
  if (fk != null) for (int i = 0; i < fk.length; i++) {
    int c = (int) Math.round((fk[i] - x0) / (x1 - x0) * (Wd - 1));
    int r = H - 1 - (int) Math.round((fv[i] - y0) / (y1 - y0) * (H - 1));
    if (r >= 0 && r < H && c >= 0 && c < Wd && g[r][c] == ' ') g[r][c] = '.';
  }
  for (int i = 0; i < k.length; i++) {
    int c = (int) Math.round((k[i] - x0) / (x1 - x0) * (Wd - 1));
    int r = H - 1 - (int) Math.round((v[i] - y0) / (y1 - y0) * (H - 1));
    if (r >= 0 && r < H && c >= 0 && c < Wd) g[r][c] = '*';
  }
  for (int r = 0; r < H; r++)
    System.out.println("   " + grey(String.format(Locale.ROOT, "%6.3f ", y1 - r * (y1 - y0) / (H - 1)))
        + grey("│") + white(new String(g[r])));
  System.out.println("          " + grey("└" + "─".repeat(Wd)));
  System.out.println("           " + grey(String.format(Locale.ROOT,
      "%-" + (Wd / 2) + ".0f%" + (Wd - Wd / 2) + ".0f", x0, x1)) + grey("   strike"));
}
double[] denseK() {
  double[] a = new double[60];
  for (int i = 0; i < 60; i++) a[i] = STRIKES[0] + (STRIKES[STRIKES.length - 1] - STRIKES[0]) * i / 59.0;
  return a;
}
double[] sabrVols(double[] ks, double al, double rh, double nu) {
  double[] a = new double[ks.length];
  for (int i = 0; i < ks.length; i++) {
    double k = Math.abs(ks[i] - F) < 1e-6 ? ks[i] + 1e-3 : ks[i];   // tape blackVol has no ATM branch
    a[i] = SabrHagan.blackVol(al, BETA, rh, nu, F, k, T);
  }
  return a;
}
SETUP

banner "1 · The smile on the screen"
say "Seven option quotes from the broker, one expiry, converted to Black vols."
say "The desk needs a SABR fit — three free parameters — that reprices them."

run <<'CODE'
for (int i = 0; i < STRIKES.length; i++)
    System.out.printf(Locale.ROOT, "   K = %s   vol = %s%n",
        yellow(String.format("%.0f", STRIKES[i])),
        white(String.format("%.6f", target[i])));
CODE

run <<'CODE'
plotSmile(STRIKES, target, null, null);
CODE

say "Vol dips near the money and lifts on both wings — that shape is the smile."
say "A faint left tilt — puts a shade richer than calls. One Black-Scholes"
say "number is a flat line through it; a richer model has to hold the curvature."
nap 0.6

banner "2 · The objective, recorded once"
say "Sum of squared vol errors across the seven strikes, written against the"
say "recorder. alpha, rho, nu are the inputs; beta is pinned. This tape is"
say "built one time — every L-BFGS iteration just replays it forward and back."

run <<'CODE'
Consumer<AadRecorder> objective = rec -> {
    SDouble alpha = rec.input("alpha", 0.20);
    SDouble rho   = rec.input("rho",   0.0);
    SDouble nu    = rec.input("nu",    0.30);
    SDouble beta  = rec.constant(BETA);
    SDouble sse   = rec.constant(0.0);
    for (int i = 0; i < STRIKES.length; i++) {
        SDouble model = SabrHagan.blackVol(rec, alpha, beta, rho, nu, F, STRIKES[i], T);
        SDouble diff  = model.sub(target[i]);
        sse = sse.add(diff.mul(diff));
    }
    rec.output(sse);
};
CODE

banner "3 · Solve"
note "adjoint-gradient L-BFGS: one reverse sweep gives d(SSE)/d(alpha,rho,nu) at once"

run <<'CODE'
var cal = Calibrator.of(objective);
cal = cal.parameter("alpha", 0.30, 1e-4, 2.0);
cal = cal.parameter("rho", 0.0, -0.999, 0.999);
cal = cal.parameter("nu", 0.50, 1e-4, 5.0);
cal = cal.maxIterations(80).tolerance(1e-12);
long t0 = now();
var r = cal.solve();
double ms = msSince(t0);
CODE

run <<'CODE'
System.out.printf(Locale.ROOT, "   target    : alpha %s  rho %s  nu %s%n",
    white("0.2000"), white("-0.0500"), white("1.0000"));
System.out.printf(Locale.ROOT, "   recovered : alpha %s  rho %s  nu %s%n",
    green(String.format("%.4f", r.parameters().get("alpha"))),
    green(String.format("%+.4f", r.parameters().get("rho"))),
    green(String.format("%.4f", r.parameters().get("nu"))));
System.out.printf(Locale.ROOT, "   residual SSE %s   %s iterations   %s ms   converged=%s%n",
    yellow(String.format("%.2e", r.objective())),
    white(r.iterations()), yellow(String.format("%.0f", ms)), white(r.converged()));
CODE

banner "4 · Does it reprice the smile?"

run <<'CODE'
double a = r.parameters().get("alpha"), rh = r.parameters().get("rho"), n = r.parameters().get("nu");
double[] fk = denseK();
plotSmile(STRIKES, target, fk, sabrVols(fk, a, rh, n));
CODE

say "The dotted line is the fitted SABR curve; it lands on every quote."
nap 0.5

run <<'CODE'
System.out.println(cyan("   strike     target vol     fitted vol"));
for (double k : STRIKES)
    System.out.printf(Locale.ROOT, "   %s   %s   %s%n",
        yellow(String.format("%.0f", k)),
        white(String.format("%.6f", SabrHagan.blackVol(TRUE_ALPHA, BETA, TRUE_RHO, TRUE_NU, F, k, T))),
        white(String.format("%.6f", SabrHagan.blackVol(a, BETA, rh, n, F, k, T))));
CODE

say "Exact to the printed digits. No finite-difference gradient, no step size,"
say "no hand-coded derivative of the Hagan formula — the reverse sweep did it."

finale "The gradient of the calibration objective, for free, every step." \
       "Swap SABR for Heston, or the objective for a full MC repricing." \
       "See also: demo/greeks-on-gpu.sh"
