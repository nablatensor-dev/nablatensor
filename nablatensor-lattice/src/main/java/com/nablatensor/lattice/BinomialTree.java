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
package com.nablatensor.lattice;

import com.nablatensor.lattice.LatticePayoff.ExerciseSchedule;
import com.nablatensor.quant.OptionType;

/**
 * A recombining binomial tree with backward induction — the method the
 * curriculum introduces risk-neutral valuation and early exercise with, and the
 * one thing a record-and-replay Monte-Carlo engine cannot do. Deliberately not
 * on the adjoint tape: this is a plain {@code double}, {@code O(n^2)} companion.
 *
 * <p>Three lattice parameterisations:
 * <ul>
 *   <li>{@link Method#CRR} — Cox-Ross-Rubinstein, {@code u = e^{sigma sqrt(dt)}};
 *       {@code O(1/n)} convergence with the familiar even/odd oscillation;</li>
 *   <li>{@link Method#JARROW_RUDD} — equal-probability;</li>
 *   <li>{@link Method#LEISEN_REIMER} — Peizer-Pratt inversion of the
 *       Black-Scholes {@code d1}, {@code d2}; smooth {@code O(1/n^2)}
 *       convergence, odd step count, vanilla only.</li>
 * </ul>
 */
public final class BinomialTree {

  public enum Method { CRR, JARROW_RUDD, LEISEN_REIMER }

  private final double spot;
  private final double rate;
  private final double dividendYield;
  private final double vol;
  private final double maturity;
  private final int steps;
  private final Method method;

  private double[] slice1;    // values at time dt
  private double[] slice2;    // values at time 2 dt

  private BinomialTree(double spot, double rate, double dividendYield, double vol,
                       double maturity, int steps, Method method) {
    this.spot = spot;
    this.rate = rate;
    this.dividendYield = dividendYield;
    this.vol = vol;
    this.maturity = maturity;
    this.steps = method == Method.LEISEN_REIMER && steps % 2 == 0 ? steps + 1 : steps;
    this.method = method;
  }

  public static BinomialTree of(double spot, double rate, double dividendYield, double vol,
                                double maturity, int steps, Method method) {
    return new BinomialTree(spot, rate, dividendYield, vol, maturity, steps, method);
  }

  public int steps() {
    return steps;
  }

  /**
   * Price a general payoff under an exercise schedule ({@link Method#CRR} or
   * {@link Method#JARROW_RUDD}); the terminal payoff also serves as the
   * early-exercise payoff.
   */
  public double price(LatticePayoff payoff, ExerciseSchedule schedule) {
    if (method == Method.LEISEN_REIMER) {
      throw new IllegalStateException("Leisen-Reimer is vanilla-only; use priceVanilla");
    }
    Lattice l = lattice(0.0);
    return induct(payoff, payoff, schedule, l);
  }

  /** Price a vanilla call/put; the only entry point that supports Leisen-Reimer. */
  public double priceVanilla(OptionType type, double strike, ExerciseSchedule schedule) {
    LatticePayoff payoff = LatticePayoff.vanilla(type, strike);
    Lattice l = lattice(strike);
    return induct(payoff, payoff, schedule, l);
  }

  // ---- lattice construction --------------------------------------------

  private record Lattice(double u, double d, double p, double disc) {}

  private Lattice lattice(double strikeForLr) {
    int n = steps;
    double dt = maturity / n;
    double disc = Math.exp(-rate * dt);
    double growth = Math.exp((rate - dividendYield) * dt);
    switch (method) {
      case CRR -> {
        double u = Math.exp(vol * Math.sqrt(dt));
        return new Lattice(u, 1.0 / u, (growth - 1.0 / u) / (u - 1.0 / u), disc);
      }
      case JARROW_RUDD -> {
        double nu = (rate - dividendYield - 0.5 * vol * vol) * dt;
        double sd = vol * Math.sqrt(dt);
        return new Lattice(Math.exp(nu + sd), Math.exp(nu - sd), 0.5, disc);
      }
      case LEISEN_REIMER -> {
        double sqrtT = Math.sqrt(maturity);
        double d1 = (Math.log(spot / strikeForLr)
            + (rate - dividendYield + 0.5 * vol * vol) * maturity) / (vol * sqrtT);
        double d2 = d1 - vol * sqrtT;
        double p = peizerPratt(d2, n);
        double pStar = peizerPratt(d1, n);
        double u = growth * pStar / p;
        double d = (growth - p * u) / (1.0 - p);
        return new Lattice(u, d, p, disc);
      }
      default -> throw new IllegalStateException();
    }
  }

  private double induct(LatticePayoff terminal, LatticePayoff early, ExerciseSchedule schedule, Lattice l) {
    int n = steps;
    double[] v = new double[n + 1];
    for (int j = 0; j <= n; j++) {
      v[j] = terminal.exerciseValue(nodeSpot(l, n, j));
    }
    slice1 = null;
    slice2 = null;
    for (int i = n - 1; i >= 0; i--) {
      double[] next = new double[i + 1];
      boolean exercisable = schedule.exercisableAtStep(i, n);
      for (int j = 0; j <= i; j++) {
        double cont = l.disc() * (l.p() * v[j + 1] + (1.0 - l.p()) * v[j]);
        if (exercisable) {
          cont = Math.max(cont, early.exerciseValue(nodeSpot(l, i, j)));
        }
        next[j] = cont;
      }
      if (i == 2) {
        slice2 = next.clone();
      }
      if (i == 1) {
        slice1 = next.clone();
      }
      v = next;
    }
    return v[0];
  }

  private double nodeSpot(Lattice l, int step, int j) {
    return spot * Math.pow(l.u(), j) * Math.pow(l.d(), step - j);
  }

  // ---- tree Greeks helpers -------------------------------------------

  double[] slice1() {
    return slice1 == null ? null : slice1.clone();
  }

  double[] slice2() {
    return slice2 == null ? null : slice2.clone();
  }

  double spot() {
    return spot;
  }

  double[] nodeSpots1() {
    Lattice l = lattice(spot);   // strike unused for CRR/JR; ok for LR node spots too
    return new double[] {nodeSpot(l, 1, 0), nodeSpot(l, 1, 1)};
  }

  double[] nodeSpots2() {
    Lattice l = lattice(spot);
    return new double[] {nodeSpot(l, 2, 0), nodeSpot(l, 2, 1), nodeSpot(l, 2, 2)};
  }

  private static double peizerPratt(double z, int n) {
    double nOdd = n % 2 == 0 ? n + 1 : n;
    double c = z / (nOdd + 1.0 / 3.0 + 0.1 / (nOdd + 1.0));
    double root = Math.sqrt(1.0 - Math.exp(-(c * c) * (nOdd + 1.0 / 6.0)));
    return 0.5 + Math.copySign(0.5 * root, z);
  }
}
