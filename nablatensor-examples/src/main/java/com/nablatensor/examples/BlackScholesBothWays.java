/*
 * Copyright 2026 The NablaTensor Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nablatensor.examples;

import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.BlackScholes;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.OptionType;
import com.nablatensor.quant.Products;
import com.nablatensor.quant.analytic.GeneralizedBsm;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.UnaryOperator;

/**
 * Black-Scholes, both ways — the {@code main()} counterpart of
 * {@code demo/black-scholes-both-ways.sh}, without the narration and with the
 * timing done carefully.
 *
 * <ol>
 *   <li>the Black-Scholes-Merton closed form;</li>
 *   <li>the same European priced by a large adjoint Monte-Carlo run, value plus
 *       every first-order Greek from one reverse sweep, checked against the
 *       closed form;</li>
 *   <li>a convergence ladder: one run per size, the gap measured against the
 *       standard error the closed form itself predicts for that many paths;</li>
 *   <li>adjoint vs. bump-and-revalue, <em>measured properly</em> — a GPU's
 *       clock ramps under load, so both sides are warmed and the <b>best</b> of
 *       several runs is taken, not the mean of a cold handful. Measured this way
 *       the reverse sweep for {@code value + 5 Greeks} costs about the same as a
 *       single price pass, so it beats the eleven-pass central difference by
 *       roughly the replay-count ratio;</li>
 *   <li>an implied-vol solve whose vega is one component of that same reverse
 *       sweep — Newton in a handful of steps vs. a derivative-free bisection;</li>
 *   <li>put-call parity and a continuous dividend yield off the closed form.</li>
 * </ol>
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.BlackScholesBothWays}
 * (optionally {@code -Dscenarios=... -Dengine=vulkan|cuda|cpu-jit}).
 */
public final class BlackScholesBothWays {

  private BlackScholesBothWays() {
  }

  public static void main(String[] args) {
    EquityMarket market = EquityMarket.atmOneYear();
    long seed = Long.getLong("seed", 42L);
    String engine = System.getProperty("engine", "");

    BlackScholes bs = BlackScholes.of(OptionType.CALL, market);
    System.out.printf(Locale.ROOT, "1 . Black-Scholes-Merton closed form (S0=K=100, sigma=20%%, r=3%%, T=1y)%n");
    System.out.printf(Locale.ROOT,
        "    price %.5f   delta %.5f   vega %.4f   rho %.4f   dV/dK %.5f%n%n",
        bs.price(), bs.delta(), bs.vega(), bs.rho(), bs.strikeSensitivity());

    try (MonteCarlo<EquityMarket> mc = build(market, engine, true);
         MonteCarlo<EquityMarket> po = build(market, engine, false)) {

      String eng = mc.engine();
      boolean gpu = !eng.equals("cpu-jit") && !eng.equals("cpu") && !eng.equals("simd");
      long n = Long.getLong("scenarios", gpu ? 100_000_000L : 5_000_000L);

      // 2 . the same option, simulated -----------------------------------
      Nabla.TypedValuation<EquityMarket> p = mc.run(n, seed);
      EquityMarket g = p.greeks();
      System.out.printf(Locale.ROOT, "2 . Adjoint Monte-Carlo on %s  (%d tape nodes)%n", eng, mc.nodes());
      System.out.printf(Locale.ROOT, "    %-7s %15s %15s %13s%n", "", "adjoint MC", "closed form", "|diff|");
      row("price", p.price(), bs.price());
      row("delta", g.spot(), bs.delta());
      row("vega", g.vol(), bs.vega());
      row("rho", g.rate(), bs.rho());
      row("dV/dK", g.strike(), bs.strikeSensitivity());
      System.out.printf(Locale.ROOT, "    %,d paths in %.1f ms  =  %.2e paths/s%n%n",
          p.scenarios(), p.seconds() * 1e3, p.scenariosPerSecond());

      // 3 . convergence onto the formula -------------------------------
      System.out.printf(Locale.ROOT,
          "3 . Convergence -- one run per size, gap vs. the analytic standard error%n");
      System.out.printf(Locale.ROOT, "    %14s %15s %13s %9s%n", "paths", "MC price", "|error|", "err/SE");
      for (long m : ladder(n, gpu)) {
        double price = mc.run(m, seed).price();
        double err = Math.abs(price - bs.price());
        System.out.printf(Locale.ROOT, "    %14d %15.5f %13.2e %9.2f%n",
            m, price, err, err / callStdErr(market, bs.price(), m));
      }
      System.out.println();

      // 4 . adjoint vs. bump, measured properly ---------------------
      double bumpDelta = bumpAll(po, market, n);          // also the first warm-up pass
      warm(8, () -> mc.run(n, seed));
      warm(6, () -> bumpAll(po, market, n));
      double adjMs = best(15, () -> mc.run(n, seed));
      double bumpMs = best(6, () -> bumpAll(po, market, n));
      System.out.printf(Locale.ROOT,
          "4 . Adjoint vs. bump-and-revalue  (warmed, best of N -- a GPU's clock ramps)%n");
      System.out.printf(Locale.ROOT, "    bump delta %.5f   adjoint %.5f   closed form %.5f%n",
          bumpDelta, g.spot(), bs.delta());
      System.out.printf(Locale.ROOT,
          "    value + 5 Greeks: 1 reverse sweep %.2f ms   vs   11 price passes %.2f ms%n", adjMs, bumpMs);
      System.out.printf(Locale.ROOT,
          "    speedup %.1fx   (one sweep, any number of Greeks; bump grows +2 passes per factor)%n%n",
          bumpMs / adjMs);

      // 5 . implied volatility, vega from the sweep ----------------
      double quote = BlackScholes.of(OptionType.CALL, market.withVol(0.28)).price();
      long ivN = Math.min(n, gpu ? 20_000_000L : 2_000_000L);
      double sigma = 0.15;
      int newton = 0;
      for (; newton < 12; newton++) {
        Nabla.TypedValuation<EquityMarket> v = mc.run(market.withVol(sigma), ivN, seed);
        double f = v.price() - quote;
        if (Math.abs(f) < 1e-5) {
          break;
        }
        sigma = clamp(sigma - f / v.greeks().vol(), 0.01, 2.0);
      }
      int bisect = 0;
      double lo = 0.05;
      double hi = 1.0;
      while (hi - lo > 1e-6) {
        double mid = 0.5 * (lo + hi);
        if (mc.run(market.withVol(mid), ivN, seed).price() - quote > 0) {
          hi = mid;
        } else {
          lo = mid;
        }
        bisect++;
      }
      System.out.printf(Locale.ROOT, "5 . Implied volatility (quote %.5f, true vol 0.2800)%n", quote);
      System.out.printf(Locale.ROOT,
          "    Newton, vega from the sweep: %d steps -> %.6f     bisection: %d steps -> %.6f%n%n",
          newton, sigma, bisect, 0.5 * (lo + hi));

      // 6 . identities off the closed form ------------------------
      BlackScholes put = BlackScholes.of(OptionType.PUT, market);
      double fwd = market.strike() * Math.exp(-market.rate() * market.maturity());
      double parity = bs.price() - put.price() - (market.spot() - fwd);
      double q2 = GeneralizedBsm.of(OptionType.CALL, 100.0, 100.0, 1.0, 0.03, 0.02, 0.20).price();
      System.out.printf(Locale.ROOT, "6 . Checks%n");
      System.out.printf(Locale.ROOT,
          "    put-call parity residual %.2e     dividend yield q=2%%: price %.5f  (q=0: %.5f)%n",
          parity, q2, bs.price());
    }
  }

  private static MonteCarlo<EquityMarket> build(EquityMarket market, String engine, boolean greeks) {
    MonteCarlo.Builder<EquityMarket> b =
        MonteCarlo.of(Products.europeanCall()).market(market).steps(1).fp32();
    b = greeks ? b.greeks() : b.priceOnly();
    b = engine.isEmpty() ? b.fastest() : b.on(engine);
    return b.build();
  }

  /** Sizes for the convergence ladder: the fixed decades below {@code n}, then {@code n} itself. */
  private static long[] ladder(long n, boolean gpu) {
    long[] decades = gpu
        ? new long[] {10_000L, 100_000L, 1_000_000L, 10_000_000L}
        : new long[] {10_000L, 100_000L, 1_000_000L};
    List<Long> out = new ArrayList<>();
    for (long d : decades) {
      if (d < n) {
        out.add(d);
      }
    }
    out.add(n);
    return out.stream().mapToLong(Long::longValue).toArray();
  }

  /** The whole bump-and-revalue set: base + two price replays per input = 11 passes. Returns bump delta. */
  private static double bumpAll(MonteCarlo<EquityMarket> po, EquityMarket m, long n) {
    double s = m.spot();
    double k = m.strike();
    double v = m.vol();
    po.run(n, 42L);
    double d = cdiff(po, m, n, x -> x.withSpot(s * 1.005), x -> x.withSpot(s * 0.995), s * 0.01);
    cdiff(po, m, n, x -> x.withVol(v + 1e-4), x -> x.withVol(v - 1e-4), 2e-4);
    cdiff(po, m, n, x -> x.withRate(0.0301), x -> x.withRate(0.0299), 2e-4);
    cdiff(po, m, n, x -> x.withStrike(k * 1.005), x -> x.withStrike(k * 0.995), k * 0.01);
    cdiff(po, m, n, x -> x.withMaturity(1.001), x -> x.withMaturity(0.999), 2e-3);
    return d;
  }

  private static double cdiff(MonteCarlo<EquityMarket> po, EquityMarket m, long n,
      UnaryOperator<EquityMarket> up, UnaryOperator<EquityMarket> down, double h) {
    double hi = po.run(up.apply(m), n, 42L).price();
    double lo = po.run(down.apply(m), n, 42L).price();
    return (hi - lo) / h;
  }

  private static void warm(int iters, Runnable r) {
    for (int i = 0; i < iters; i++) {
      r.run();
    }
  }

  private static double best(int reps, Runnable r) {
    double best = Double.MAX_VALUE;
    for (int i = 0; i < reps; i++) {
      long t = System.nanoTime();
      r.run();
      best = Math.min(best, (System.nanoTime() - t) / 1e6);
    }
    return best;
  }

  private static double clamp(double x, double lo, double hi) {
    return Math.max(lo, Math.min(hi, x));
  }

  private static void row(String name, double mc, double ref) {
    System.out.printf(Locale.ROOT, "    %-7s %15.6f %15.6f %13.2e%n", name, mc, ref, Math.abs(mc - ref));
  }

  /**
   * Standard error of an {@code n}-path discounted European-call Monte-Carlo
   * estimate, from the closed form: {@code e^{-rT} sqrt(Var[(S_T-K)^+] ) / sqrt(n)}
   * with the second moment {@code E[((S_T-K)^+)^2]} in closed form for the
   * lognormal {@code S_T}.
   */
  private static double callStdErr(EquityMarket m, double bsPrice, long n) {
    double s = m.spot();
    double k = m.strike();
    double t = m.maturity();
    double r = m.rate();
    double v = m.vol();
    double disc = Math.exp(-r * t);
    double srt = v * Math.sqrt(t);
    double d1 = (Math.log(s / k) + (r + 0.5 * v * v) * t) / srt;
    double d2 = d1 - srt;
    double e2 = s * s * Math.exp((2 * r + v * v) * t) * BlackScholes.N(d1 + srt)
        - 2 * k * s * Math.exp(r * t) * BlackScholes.N(d1)
        + k * k * BlackScholes.N(d2);
    double undisc = Math.exp(r * t) * bsPrice;
    double var = disc * disc * (e2 - undisc * undisc);
    return Math.sqrt(var / n);
  }
}
