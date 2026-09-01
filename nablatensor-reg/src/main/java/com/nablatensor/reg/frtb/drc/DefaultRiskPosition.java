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
package com.nablatensor.reg.frtb.drc;

import com.nablatensor.reg.frtb.drc.DrcParameters.CreditQuality;
import com.nablatensor.reg.frtb.drc.DrcParameters.DrcBucket;
import com.nablatensor.reg.frtb.drc.DrcParameters.Seniority;

/**
 * One position feeding the Default Risk Charge (MAR22). A long position has
 * {@code notional > 0}, a short {@code notional < 0}. The gross jump-to-default
 * amount is {@code LGD * notional + (marketValue - notional)} (MAR22.8),
 * maturity-scaled per {@link Jtd}.
 *
 * @param obligor            obligor identifier — same-obligor longs and shorts net
 * @param bucket             the non-securitisation bucket (corporates / sovereigns / local government)
 * @param seniority          drives loss-given-default
 * @param quality            drives the default risk weight
 * @param notional           signed notional (long &gt; 0, short &lt; 0)
 * @param marketValue        current market value of the position (same sign convention)
 * @param maturityYears      residual maturity in years (for the &lt; 1yr scaling)
 */
public record DefaultRiskPosition(String obligor, DrcBucket bucket, Seniority seniority,
                                  CreditQuality quality, double notional, double marketValue,
                                  double maturityYears) {
}
