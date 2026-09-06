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
package com.nablatensor.credit;

/**
 * A synthetic CDO tranche {@code [attach, detach]} on a homogeneous pool, priced
 * off the {@link PortfolioLossDistribution}: the protection leg pays the change
 * in expected tranche loss over each period, the premium leg pays a spread on
 * the surviving (un-lost) tranche notional, and the par spread is their ratio.
 *
 * <p>Losses fall on the pool's integer default grid, so the expected tranche
 * loss at each payment date is a single sum over that grid.
 */
public record CdoTranche(double attach, double detach) {

  public CdoTranche {
    if (!(detach > attach && attach >= 0.0 && detach <= 1.0)) {
      throw new IllegalArgumentException("need 0 <= attach < detach <= 1");
    }
  }

  public double width() {
    return detach - attach;
  }

  /** Expected loss suffered by the tranche by the horizon, as a fraction of the tranche notional. */
  public double expectedLossFraction(PortfolioLossDistribution loss) {
    return loss.expectedTrancheLoss(attach, detach) / width();
  }

  /**
   * Par spread of the tranche.
   *
   * @param lossAtDate     expected portfolio-level tranche loss at each payment date (ascending)
   * @param paymentTimes   ascending payment times in years
   * @param discount       time -> risk-free discount factor
   */
  public double parSpread(double[] lossAtDate, double[] paymentTimes,
                          java.util.function.DoubleUnaryOperator discount) {
    if (lossAtDate.length != paymentTimes.length || lossAtDate.length == 0) {
      throw new IllegalArgumentException("lossAtDate and paymentTimes must be non-empty and equal length");
    }
    double w = width();
    double protection = 0.0;
    double premium = 0.0;
    double prevLoss = 0.0;
    double prevT = 0.0;
    for (int i = 0; i < paymentTimes.length; i++) {
      double t = paymentTimes[i];
      double dt = t - prevT;
      double loss = lossAtDate[i];
      double dfMid = 0.5 * (discount.applyAsDouble(prevT) + discount.applyAsDouble(t));
      protection += dfMid * (loss - prevLoss);
      // premium accrues on the average surviving tranche notional over the period
      double survivingNow = Math.max(w - loss, 0.0);
      double survivingPrev = Math.max(w - prevLoss, 0.0);
      premium += dt * discount.applyAsDouble(t) * 0.5 * (survivingNow + survivingPrev);
      prevLoss = loss;
      prevT = t;
    }
    return protection / premium;
  }

  /** PV to the protection buyer of paying {@code contractSpread} on unit tranche notional. */
  public double protectionBuyerPv(double contractSpread, double[] lossAtDate, double[] paymentTimes,
                                  java.util.function.DoubleUnaryOperator discount) {
    double par = parSpread(lossAtDate, paymentTimes, discount);
    // PV = (par - contract) * risky annuity; par*annuity == protection leg.
    double annuity = protectionLeg(lossAtDate, paymentTimes, discount) / par;
    return (par - contractSpread) * annuity;
  }

  private double protectionLeg(double[] lossAtDate, double[] paymentTimes,
                               java.util.function.DoubleUnaryOperator discount) {
    double protection = 0.0;
    double prevLoss = 0.0;
    double prevT = 0.0;
    for (int i = 0; i < paymentTimes.length; i++) {
      double dfMid = 0.5 * (discount.applyAsDouble(prevT) + discount.applyAsDouble(paymentTimes[i]));
      protection += dfMid * (lossAtDate[i] - prevLoss);
      prevLoss = lossAtDate[i];
      prevT = paymentTimes[i];
    }
    return protection;
  }
}
