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

/**
 * Geometric Brownian motion as a composable model step (Seam 5).
 *
 * <p>Each step is the exact log-Euler solution of {@code dS = rS dt + sigma S dW}
 * over that step's interval:
 *
 * <pre>{@code S_{i+1} = S_i * exp((r - sigma^2/2) dt_i + sigma sqrt(dt_i) Z)}</pre>
 *
 * <p>with {@code dt_i = maturity * grid.fraction(i)}. For a
 * {@link TimeGrid#uniform(int)} grid every {@code dt_i} is identical and the
 * per-step drift and diffusion are a single shared tape node, so the recording
 * matches the earlier fixed-{@code dt} form exactly.
 *
 * <p>Subclass and override {@link #drift(SDouble)} or {@link #diffusion(SDouble)}
 * to get a displaced-diffusion or a term-structure variant without touching the
 * driver or the engine.
 */
public class GbmPath {

  private final SDouble[] drift;      // (r - sigma^2/2) * dt_i
  private final SDouble[] diffusion;  // sigma * sqrt(dt_i)

  /** Uniform-grid convenience: {@code n} equal steps to {@code maturity}. */
  public GbmPath(AadRecorder rec, SDouble rate, SDouble vol, int steps, SDouble maturity) {
    this(rec, rate, vol, TimeGrid.uniform(steps), maturity);
  }

  public GbmPath(AadRecorder rec, SDouble rate, SDouble vol, TimeGrid grid, SDouble maturity) {
    int n = grid.steps();
    this.drift = new SDouble[n];
    this.diffusion = new SDouble[n];
    SDouble halfVar = rate.sub(vol.mul(vol).mul(0.5));
    if (grid.isUniform()) {
      SDouble dt = maturity.mul(grid.fraction(0));
      SDouble d = drift(halfVar.mul(dt));
      SDouble s = diffusion(vol.mul(dt.sqrt()));
      for (int i = 0; i < n; i++) {
        drift[i] = d;
        diffusion[i] = s;
      }
    } else {
      for (int i = 0; i < n; i++) {
        SDouble dt = maturity.mul(grid.fraction(i));
        drift[i] = drift(halfVar.mul(dt));
        diffusion[i] = diffusion(vol.mul(dt.sqrt()));
      }
    }
  }

  /** Hook: the deterministic per-step log-return. Identity for plain GBM. */
  protected SDouble drift(SDouble perStepDrift) {
    return perStepDrift;
  }

  /** Hook: the per-step volatility multiplier on {@code Z}. Identity for plain GBM. */
  protected SDouble diffusion(SDouble perStepVol) {
    return perStepVol;
  }

  /** One step forward from step {@code i} given a standard-normal draw {@code z}. */
  public SDouble step(SDouble spot, SDouble z, int i) {
    return spot.mul(drift[i].add(diffusion[i].mul(z)).exp());
  }
}
