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

import com.nablatensor.codegen.Of;

/**
 * One point on a counterparty CDS par-spread curve: a protection tenor and the
 * quoted par spread in basis points.
 *
 * @param tenorYears  protection maturity in years, ascending across a curve
 * @param parSpreadBp par CDS spread in basis points (e.g. {@code 120.0} = 120 bp)
 */
@Of
public final class CdsQuote {

  private final double tenorYears;
  private final double parSpreadBp;
  static CdsQuote create(double tenorYears, double parSpreadBp) {
    return new CdsQuote(tenorYears, parSpreadBp);
  }

  public static CdsQuoteBuilder of() {
    return new CdsQuoteBuilder();
  }

  public double tenorYears() {
    return tenorYears;
  }

  public double parSpreadBp() {
    return parSpreadBp;
  }


  private CdsQuote(double tenorYears, double parSpreadBp) {
    if (!(tenorYears > 0.0)) {
      throw new IllegalArgumentException("tenorYears must be > 0, got " + tenorYears);
    }
    if (!(parSpreadBp >= 0.0)) {
      throw new IllegalArgumentException("parSpreadBp must be >= 0, got " + parSpreadBp);
    }
    this.tenorYears = tenorYears;
    this.parSpreadBp = parSpreadBp;
  }

  public double parSpread() {
    return parSpreadBp * 1.0e-4;
  }
}
