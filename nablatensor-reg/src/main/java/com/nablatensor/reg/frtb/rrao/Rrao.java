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
package com.nablatensor.reg.frtb.rrao;

import java.util.List;

/**
 * FRTB SA <b>Residual Risk Add-On</b> (MAR23): a gross-notional surcharge for
 * risks the SBM + DRC framework does not capture.
 *
 * <pre>{@code
 * RRAO = 1.0% * Sum grossNotional(exotic underlying)
 *      + 0.1% * Sum grossNotional(other residual risk)
 * }</pre>
 *
 * No netting, no aggregation formula — a plain weighted sum.
 */
public final class Rrao {

  /** Which MAR23 residual-risk tier an instrument falls in. */
  public enum ResidualRiskKind {
    /** Exotic underlying (longevity, weather, natural catastrophe, realised volatility, ...). */
    EXOTIC_UNDERLYING(0.010),
    /** Gap risk (digital / binary / barrier options, ...). */
    GAP_RISK(0.001),
    /** Correlation risk (basket options, CDO tranches, best-/worst-of, ...). */
    CORRELATION_RISK(0.001),
    /** Behavioural risk (callable / putable bonds, non-vanilla prepayment, ...). */
    BEHAVIOURAL_RISK(0.001),
    /** Dividend risk (payoff depends on realised dividends other than a known cash flow). */
    DIVIDEND_RISK(0.001),
    /** Any other residual risk in scope of MAR23. */
    OTHER(0.001);

    private final double riskWeight;

    ResidualRiskKind(double riskWeight) {
      this.riskWeight = riskWeight;
    }

    /** The MAR23 risk weight as a decimal fraction (1.0% or 0.1%). */
    public double riskWeight() {
      return riskWeight;
    }
  }

  /**
   * One instrument bearing residual risk.
   *
   * @param id            instrument identifier (informational)
   * @param grossNotional gross notional (always taken as a magnitude)
   * @param kind          the residual-risk tier
   */
  public record ResidualRiskPosition(String id, double grossNotional, ResidualRiskKind kind) {
  }

  /** The RRAO components and total. */
  public record Result(double exotic, double other, double total) {
  }

  private final List<ResidualRiskPosition> positions;

  private Rrao(List<ResidualRiskPosition> positions) {
    this.positions = List.copyOf(positions);
  }

  public static Rrao of(List<ResidualRiskPosition> positions) {
    return new Rrao(positions);
  }

  public Result compute() {
    double exotic = 0.0;
    double other = 0.0;
    for (ResidualRiskPosition p : positions) {
      double add = p.kind().riskWeight() * Math.abs(p.grossNotional());
      if (p.kind() == ResidualRiskKind.EXOTIC_UNDERLYING) {
        exotic += add;
      } else {
        other += add;
      }
    }
    return new Result(exotic, other, exotic + other);
  }

  /** The RRAO total for a set of positions. */
  public static double total(List<ResidualRiskPosition> positions) {
    return of(positions).compute().total();
  }
}
