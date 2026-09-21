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

import com.nablatensor.quant.OptionTypeEnum;

/**
 * Garman-Kohlhagen — a European option on an FX rate quoted as units of
 * domestic per unit of foreign. The foreign rate {@code r_f} acts as a
 * continuous yield on the underlying, so the cost of carry is {@code b = r_d - r_f}
 * and discounting is at the domestic rate {@code r_d}.
 *
 * <p>{@link AnalyticGreeks#rho()} is {@code dV/dr_d} (domestic rho, carry
 * tracking); {@link #foreignRho} is {@code dV/dr_f}.
 */
public final class GarmanKohlhagen {

  private final AnalyticGreeks greeks;
  private final double foreignRho;

  private GarmanKohlhagen(AnalyticGreeks greeks, double foreignRho) { this.greeks = greeks; this.foreignRho = foreignRho; }
  public AnalyticGreeks greeks() { return greeks; }
  public double foreignRho() { return foreignRho; }

  /**
   * @param type       call or put on the FX rate
   * @param spot       spot FX rate {@code X} (domestic per foreign)
   * @param strike     strike {@code K}
   * @param maturity   time to expiry in years {@code T}
   * @param rateDom    domestic continuously-compounded rate {@code r_d}
   * @param rateForeign foreign continuously-compounded rate {@code r_f}
   * @param vol        lognormal volatility of the FX rate {@code sigma}
   */
  private static GarmanKohlhagen calculate(OptionTypeEnum type, double spot, double strike, double maturity,
                                   double rateDom, double rateForeign, double vol) {
    double b = rateDom - rateForeign;
    Greeking.Price5 f = (x, k, t, rd, v) -> CostOfCarry.price(type, x, k, t, rd, rd - rateForeign, v);
    AnalyticGreeks g = (maturity <= 0.0 || vol <= 0.0)
        ? AnalyticGreeks.intrinsic(CostOfCarry.price(type, spot, strike, maturity, rateDom, b, vol))
        : Greeking.central(f, spot, strike, maturity, rateDom, vol);
    double foreignRho = -CostOfCarry.carryRho(type, spot, strike, maturity, rateDom, b, vol);
    return new GarmanKohlhagen(g, foreignRho);
  }


  public static Builder of() { return new Builder(); }

  public static final class Builder {
    private OptionTypeEnum type;
    private Double spot, strike, maturity, rateDom, rateForeign, vol;
    private Builder() {}
    public Builder type(OptionTypeEnum v) { type=v; return this; }
    public Builder spot(double v) { spot=v; return this; }
    public Builder strike(double v) { strike=v; return this; }
    public Builder maturity(double v) { maturity=v; return this; }
    public Builder rateDom(double v) { rateDom=v; return this; }
    public Builder rateForeign(double v) { rateForeign=v; return this; }
    public Builder vol(double v) { vol=v; return this; }
    public GarmanKohlhagen build() { return calculate(req(type,"type"),req(spot,"spot"),req(strike,"strike"),
        req(maturity,"maturity"),req(rateDom,"rateDom"),req(rateForeign,"rateForeign"),req(vol,"vol")); }
  }
  private static <T> T req(T v, String n) { if (v == null) throw new IllegalStateException("Required field " + n + " is not set"); return v; }

  public double price() {
    return greeks.price();
  }
}
