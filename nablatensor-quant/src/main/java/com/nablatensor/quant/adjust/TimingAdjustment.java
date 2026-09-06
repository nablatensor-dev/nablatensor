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
package com.nablatensor.quant.adjust;

/**
 * The timing adjustment: the correction to a forward rate when its payment is
 * moved away from the accrual-period end it naturally pays at.
 *
 * <p>For a LIBOR-style rate {@code L0} fixing at {@code T} and naturally paying
 * at {@code T + tau}, shifting the payment to {@code payTime} multiplies the
 * in-arrears adjustment by {@code (payTime - (T + tau)) / (-tau)} — so a payment
 * pulled forward to the fixing date recovers the full in-arrears adjustment, and
 * a payment pushed further out flips its sign.
 */
public final class TimingAdjustment {

  private TimingAdjustment() {
  }

  /**
   * @param forward        forward rate {@code L0}
   * @param blackVol       lognormal vol of the forward rate
   * @param accrual        natural accrual fraction {@code tau}
   * @param fixingTime     fixing time {@code T}
   * @param payTime        the actual payment time (natural is {@code fixingTime + accrual})
   */
  public static Adjustment liborPaymentShift(double forward, double blackVol, double accrual,
                                             double fixingTime, double payTime) {
    double naturalPay = fixingTime + accrual;
    double inArrears = ConvexityAdjustment.inArrears(forward, blackVol, accrual, fixingTime).adjustment();
    // in-arrears is the payTime == fixingTime case; scale linearly in the payment offset.
    double scale = (naturalPay - payTime) / accrual;
    return new Adjustment(forward, inArrears * scale);
  }
}
