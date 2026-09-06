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

import com.nablatensor.quant.OptionType;
import com.nablatensor.quant.QuantoMarket;
import com.nablatensor.quant.adjust.Adjustment;
import com.nablatensor.quant.adjust.ConvexityAdjustment;
import com.nablatensor.quant.adjust.QuantoAdjustment;
import com.nablatensor.quant.adjust.TimingAdjustment;
import java.util.Locale;

/**
 * Feature F8 — the measure-mismatch corrections: Eurodollar futures convexity,
 * LIBOR in arrears, CMS convexity, payment timing, and the quanto adjustment.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.ConvexityQuantoShowcase}
 */
public final class ConvexityQuantoShowcase {

  private ConvexityQuantoShowcase() {
  }

  public static void main(String[] args) {
    System.out.printf(Locale.ROOT, "Convexity / timing / quanto adjustments%n%n");

    // Eurodollar future: 2y expiry, 3M rate, Hull-White a=0.03, sigma=90bp.
    ConvexityAdjustment.FuturesConvexity ed =
        ConvexityAdjustment.eurodollarFutures(0.03, 0.009, 2.0, 2.25);
    System.out.printf(Locale.ROOT, "  Eurodollar futures (2y, 3M):   %.2f bp   forward = futures - adj%n",
        ed.rateAdjustment() * 1e4);
    System.out.printf(Locale.ROOT, "     d(adj)/d(sigma) = %.4f   d(adj)/d(a) = %.5f%n",
        ed.dSigma(), ed.dMeanReversion());

    // LIBOR in arrears: 3y fixing, semi-annual, 3% forward, 20% Black vol.
    Adjustment arr = ConvexityAdjustment.inArrears(0.03, 0.20, 0.5, 3.0);
    System.out.printf(Locale.ROOT, "  LIBOR in arrears (3y):         %.2f bp   -> %.4f%%%n",
        arr.adjustmentBp(), 100 * arr.adjustedRate());

    // Same rate paid one accrual early vs one accrual late.
    Adjustment early = TimingAdjustment.liborPaymentShift(0.03, 0.20, 0.5, 3.0, 3.0);
    Adjustment late = TimingAdjustment.liborPaymentShift(0.03, 0.20, 0.5, 3.0, 4.0);
    System.out.printf(Locale.ROOT, "  timing: pay at fixing %.2f bp   |   pay one period late %.2f bp%n",
        early.adjustmentBp(), late.adjustmentBp());

    // CMS: 10y swap rate observed at 5y, semi-annual, 25% vol.
    Adjustment cms = ConvexityAdjustment.cms(0.032, 0.25, 5.0, 2, 20);
    System.out.printf(Locale.ROOT, "  CMS 10y rate at 5y:            %.2f bp   -> %.4f%%%n",
        cms.adjustmentBp(), 100 * cms.adjustedRate());

    // Quanto: foreign equity option settled in domestic currency at a fixed FX.
    QuantoMarket m = new QuantoMarket(100.0, 100.0, 0.22, 0.09, -0.35, 0.03, 0.012);
    double fwdNoQuanto = m.assetSpot() * Math.exp(m.rateForeign() * 1.0);
    double fwdQuanto = QuantoAdjustment.quantoForward(m, 1.0);
    System.out.printf(Locale.ROOT, "%n  quanto drift adjustment = %+.5f  (-rho volS volX)%n",
        QuantoAdjustment.driftAdjustment(m.corr(), m.volAsset(), m.volFx()));
    System.out.printf(Locale.ROOT, "  1y forward: no-quanto %.4f  ->  quanto %.4f%n", fwdNoQuanto, fwdQuanto);
    System.out.printf(Locale.ROOT, "  1y ATM quanto call (fixedFx=1.25):  %.5f  (delta %.4f, vega %.4f)%n",
        QuantoAdjustment.quantoOption(OptionType.CALL, m, 1.0, 1.25).price(),
        QuantoAdjustment.quantoOption(OptionType.CALL, m, 1.0, 1.25).delta(),
        QuantoAdjustment.quantoOption(OptionType.CALL, m, 1.0, 1.25).vega());
  }
}
