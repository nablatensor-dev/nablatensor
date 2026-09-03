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

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.SDouble;

/**
 * One trade in a netting set, able to mark itself to market on a simulated path
 * at any grid date. The MtM is written in plain {@link SDouble} arithmetic so it
 * records onto the same tape as the exposure simulation and one adjoint sweep
 * differentiates the whole netting-set CVA.
 *
 * <p>The set is {@code sealed}: {@link InterestRateSwap} and {@link FxForward}
 * are the two Phase-2 instrument types. Add a permit and an implementation to
 * cover another product; nothing else in the module changes.
 */
public sealed interface CvaTrade permits InterestRateSwap, FxForward {

  /** Identifier, unique within a netting set. */
  String id();

  /** Gross notional in reporting currency, for the residual-risk and BA-CVA inputs. */
  double grossNotional();

  /** Latest cash-flow date in years — the trade's contribution to netting-set maturity. */
  double effectiveMaturityYears();

  /** On-tape mark-to-market at time {@code t}, in reporting currency, on the given path. */
  SDouble markToMarket(Path path, double t);

  /** The simulated market a trade reads to value itself at a grid date. */
  interface Path {

    AadRecorder recorder();

    /** The short-rate model block, for analytic {@code P(t, T)} reconstruction. */
    HwShortRate rates();

    /** The simulated short rate at the current grid date. */
    SDouble shortRate();

    /** The simulated FX spot (reporting currency per unit foreign) at the current grid date. */
    SDouble fxSpot();

    /** {@code exp(-fxForeignRate * (to - from))} — the flat foreign-currency discount. */
    SDouble foreignDiscount(double from, double to);
  }
}
