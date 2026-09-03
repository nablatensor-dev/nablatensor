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
package com.nablatensor.cva;

import com.nablatensor.engine.SDouble;
import java.util.Locale;

/**
 * A vanilla fixed-versus-floating interest-rate swap on the simulated short-rate
 * curve. At a grid date {@code t} the outstanding fixed leg is valued from
 * analytic {@code P(t, T_j)} bonds and the floating leg is taken at par from the
 * next reset, so
 *
 * <pre>{@code
 * receiveFixedValue(t) = notional * ( fixedRate * annuity(t) - (P(t, reset) - P(t, end)) )
 * }</pre>
 *
 * and the payer swap is its negative. Only cash flows strictly after {@code t}
 * are counted, so the value amortises to zero and the netting-set exposure has
 * the usual mid-life hump.
 *
 * @param id           trade id
 * @param side         receive-fixed or pay-fixed
 * @param notional     notional in reporting currency
 * @param fixedRate    annualised fixed coupon
 * @param startYears   first accrual start in years ({@code 0} for a spot-start swap)
 * @param maturityYears final payment date in years
 * @param accrualYears payment period in years (e.g. {@code 0.5} for semi-annual)
 */
public record InterestRateSwap(String id, Side side, double notional, double fixedRate,
                               double startYears, double maturityYears, double accrualYears)
    implements CvaTrade {

  public enum Side { RECEIVE_FIXED, PAY_FIXED }

  public InterestRateSwap {
    if (!(notional > 0.0) || !(accrualYears > 0.0) || !(maturityYears > startYears)) {
      throw new IllegalArgumentException("need notional>0, accrualYears>0, maturityYears>startYears");
    }
  }

  public static InterestRateSwap payer(String id, double notional, double fixedRate, double maturityYears) {
    return new InterestRateSwap(id, Side.PAY_FIXED, notional, fixedRate, 0.0, maturityYears, 0.5);
  }

  public static InterestRateSwap receiver(String id, double notional, double fixedRate, double maturityYears) {
    return new InterestRateSwap(id, Side.RECEIVE_FIXED, notional, fixedRate, 0.0, maturityYears, 0.5);
  }

  @Override
  public String toString() {
    return String.format(Locale.ROOT, "%-12s %-12s $%,.0fm  %.2f%% fixed  %.1fy",
        id, side, notional / 1e6, fixedRate * 100.0, maturityYears);
  }

  @Override
  public double grossNotional() {
    return notional;
  }

  @Override
  public double effectiveMaturityYears() {
    return maturityYears;
  }

  @Override
  public SDouble markToMarket(Path path, double t) {
    HwShortRate model = path.rates();
    SDouble rate = path.shortRate();
    int periods = (int) Math.round((maturityYears - startYears) / accrualYears);
    SDouble annuity = path.recorder().constant(0.0);
    SDouble endBond = null;
    SDouble resetBond = null;
    for (int j = 1; j <= periods; j++) {
      double payDate = startYears + j * accrualYears;
      if (payDate <= t + 1.0e-9) {
        continue;
      }
      if (resetBond == null) {
        resetBond = model.bond(rate, t, Math.max(payDate - accrualYears, t));
      }
      SDouble discountToPay = model.bond(rate, t, payDate);
      annuity = annuity.add(discountToPay.mul(accrualYears));
      endBond = discountToPay;
    }
    if (endBond == null) {
      return path.recorder().constant(0.0);
    }
    SDouble floatingLeg = resetBond.sub(endBond);
    SDouble receiveFixedValue = annuity.mul(fixedRate).sub(floatingLeg).mul(notional);
    return receiveFixedValue.mul(side == Side.RECEIVE_FIXED ? 1.0 : -1.0);
  }
}
