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
package com.nablatensor.quant;

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.SDouble;
import com.nablatensor.engine.Nabla;
import com.nablatensor.ops.SpecialFn;
import java.util.function.BiConsumer;

/**
 * A parametric local-volatility model (CEV form) as a step block (Seam 5):
 *
 * <pre>{@code
 * sigma_loc(S) = sigma0 * (S / S0) ^ skew
 * dS = r S dt + sigma_loc(S) S dW
 * }</pre>
 *
 * <p>{@code skew = 0} is plain GBM; {@code skew < 0} is the classic equity
 * smile. {@code sigma0} and {@code skew} are differentiable inputs, so the run
 * returns the local-vol parameter risk alongside spot / rate / strike Greeks —
 * the sensitivities a smile-calibration loop consumes.
 */
public class LocalVolModel {

  private final SDouble rate;
  private final SDouble sigma0;
  private final SDouble skew;
  private final double refSpot;
  private final double dt;
  private final double sqrtDt;

  public LocalVolModel(Nabla.Inputs<LocalVolMarket> in, double refSpot, double maturity, int steps) {
    this.rate = in.of(LocalVolMarket::rate);
    this.sigma0 = in.of(LocalVolMarket::sigma0);
    this.skew = in.of(LocalVolMarket::skew);
    this.refSpot = refSpot;
    this.dt = maturity / steps;
    this.sqrtDt = Math.sqrt(dt);
  }

  /** Local volatility at the current spot. */
  public SDouble localVol(SDouble spot) {
    return sigma0.mul(SpecialFn.pow(spot.div(refSpot), skew));
  }

  public SDouble step(AadRecorder rec, SDouble spot, SDouble z) {
    SDouble sig = localVol(spot);
    SDouble drift = rate.sub(sig.mul(sig).mul(0.5)).mul(dt);
    return spot.mul(drift.add(sig.mul(sqrtDt).mul(z)).exp());
  }

  public static BiConsumer<AadRecorder, Nabla.Inputs<LocalVolMarket>> european(
      OptionType type, double maturity, int steps) {
    return (rec, in) -> {
      LocalVolModel m = new LocalVolModel(in, LocalVolMarket.REF_SPOT, maturity, steps);
      SDouble s = in.of(LocalVolMarket::spot);
      for (int t = 0; t < steps; t++) {
        s = m.step(rec, s, rec.randn());
      }
      SDouble k = in.of(LocalVolMarket::strike);
      SDouble intrinsic = type == OptionType.CALL ? s.sub(k).max(0.0) : k.sub(s).max(0.0);
      rec.output(intrinsic.mul(in.of(LocalVolMarket::rate).neg().mul(maturity).exp()));
    };
  }
}
