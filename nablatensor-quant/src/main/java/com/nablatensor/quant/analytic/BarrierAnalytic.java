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
package com.nablatensor.quant.analytic;

import com.nablatensor.quant.OptionType;

/**
 * Reiner-Rubinstein closed forms for a single-barrier European option with
 * <em>continuous</em> monitoring, zero rebate, and a constant cost of carry
 * {@code b} and volatility {@code sigma}.
 *
 * <p>This is the exact reference for the smoothed, per-step-monitored barrier in
 * {@code ExoticProducts.barrier}: the Monte-Carlo payoff there trades a small
 * bias (set by the smoothing width and the discrete monitoring) for a
 * differentiable price, and this class is what that bias is measured against.
 * Note that a discretely monitored barrier converges to the continuous price
 * only after the Broadie-Glasserman-Kou {@code exp(0.5826 sigma sqrt(dt))}
 * shift, so a like-for-like check uses many monitoring dates.
 *
 * <p>The eight in/out x up/down x call/put combinations are assembled from the
 * standard six terms {@code A, B, C, D} (the {@code E, F} rebate terms are zero
 * here) with a call/put sign {@code phi} and an up/down sign {@code eta}.
 */
public final class BarrierAnalytic {

  /** Knock direction. */
  public enum Kind { UP_IN, UP_OUT, DOWN_IN, DOWN_OUT }

  private BarrierAnalytic() {
  }

  /**
   * @param type     call or put on the terminal underlying
   * @param kind     up/down x in/out
   * @param spot     spot {@code S}
   * @param strike   strike {@code K}
   * @param barrier  barrier level {@code H}
   * @param maturity time to expiry in years {@code T}
   * @param rate     continuously-compounded discount rate {@code r}
   * @param carry    cost of carry {@code b} (use {@code r} for a non-dividend stock)
   * @param vol      lognormal volatility {@code sigma}
   */
  public static AnalyticGreeks of(OptionType type, Kind kind, double spot, double strike, double barrier,
                                  double maturity, double rate, double carry, double vol) {
    if (maturity <= 0.0 || vol <= 0.0) {
      return AnalyticGreeks.intrinsic(price(type, kind, spot, strike, barrier, maturity, rate, carry, vol));
    }
    Greeking.Price5 f = (s, k, t, r, v) -> price(type, kind, s, k, barrier, t, r, carry, v);
    return Greeking.central(f, spot, strike, maturity, rate, vol);
  }

  /** Bare price. */
  public static double price(OptionType type, Kind kind, double spot, double strike, double barrier,
                             double maturity, double rate, double carry, double vol) {
    double s = spot;
    double k = strike;
    double h = barrier;
    double t = maturity;
    double r = rate;
    double b = carry;
    double sig = vol;

    if (t <= 0.0 || sig <= 0.0) {
      boolean alive = switch (kind) {
        case UP_IN, DOWN_IN -> false;                 // never touched => in-option worthless
        case UP_OUT -> s < h;
        case DOWN_OUT -> s > h;
      };
      double intrinsic = Math.max(type.sign() * (s * Math.exp(b * t) - k), 0.0) * Math.exp(-r * t);
      return switch (kind) {
        case UP_IN, DOWN_IN -> 0.0;
        default -> alive ? intrinsic : 0.0;
      };
    }

    // Knocked-out already / knocked-in already: reduce to the trivial value.
    boolean up = kind == Kind.UP_IN || kind == Kind.UP_OUT;
    boolean in = kind == Kind.UP_IN || kind == Kind.DOWN_IN;
    double vanilla = CostOfCarry.price(type, s, k, t, r, b, sig);
    if (up && s >= h) {
      return in ? vanilla : 0.0;
    }
    if (!up && s <= h) {
      return in ? vanilla : 0.0;
    }

    double phi = type.sign();          // +1 call, -1 put
    double eta = up ? -1.0 : 1.0;      // -1 up, +1 down

    double sqrtT = Math.sqrt(t);
    double mu = (b - 0.5 * sig * sig) / (sig * sig);
    double sigSqrtT = sig * sqrtT;

    double x1 = Math.log(s / k) / sigSqrtT + (1.0 + mu) * sigSqrtT;
    double x2 = Math.log(s / h) / sigSqrtT + (1.0 + mu) * sigSqrtT;
    double y1 = Math.log(h * h / (s * k)) / sigSqrtT + (1.0 + mu) * sigSqrtT;
    double y2 = Math.log(h / s) / sigSqrtT + (1.0 + mu) * sigSqrtT;

    double carryDisc = Math.exp((b - r) * t);
    double disc = Math.exp(-r * t);
    double powPlus = Math.pow(h / s, 2.0 * (mu + 1.0));
    double powMinus = Math.pow(h / s, 2.0 * mu);

    double a = phi * s * carryDisc * Normal.cdf(phi * x1)
        - phi * k * disc * Normal.cdf(phi * x1 - phi * sigSqrtT);
    double bb = phi * s * carryDisc * Normal.cdf(phi * x2)
        - phi * k * disc * Normal.cdf(phi * x2 - phi * sigSqrtT);
    double c = phi * s * carryDisc * powPlus * Normal.cdf(eta * y1)
        - phi * k * disc * powMinus * Normal.cdf(eta * y1 - eta * sigSqrtT);
    double d = phi * s * carryDisc * powPlus * Normal.cdf(eta * y2)
        - phi * k * disc * powMinus * Normal.cdf(eta * y2 - eta * sigSqrtT);

    boolean strikeAboveBarrier = k >= h;

    double inValue;
    if (type == OptionType.CALL) {
      if (!up) { // down-and-in call
        inValue = strikeAboveBarrier ? c : a - bb + d;
      } else {   // up-and-in call
        inValue = strikeAboveBarrier ? a : bb - c + d;
      }
    } else {
      if (up) {  // up-and-in put
        inValue = strikeAboveBarrier ? a - bb + d : c;
      } else {   // down-and-in put
        inValue = strikeAboveBarrier ? bb - c + d : a;
      }
    }

    if (in) {
      return Math.max(inValue, 0.0);
    }
    // Parity: knock-in + knock-out = the vanilla (zero rebate).
    return Math.max(vanilla - inValue, 0.0);
  }
}
