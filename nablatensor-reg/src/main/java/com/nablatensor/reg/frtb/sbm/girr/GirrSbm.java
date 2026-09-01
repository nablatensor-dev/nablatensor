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
package com.nablatensor.reg.frtb.sbm.girr;

import com.nablatensor.reg.frtb.sbm.CurvatureRepricing;
import com.nablatensor.reg.frtb.sbm.SbmCharge;
import com.nablatensor.risk.Sensitivities;
import java.util.List;

/**
 * FRTB SA-SBM for the GIRR risk class — a thin binding of {@link SbmCharge} to
 * {@link GirrSbmParameters}. Sensitivities are keyed by
 * {@link com.nablatensor.risk.RiskFactor#girrDelta} /
 * {@link com.nablatensor.risk.RiskFactor#girrVega} factors; one
 * {@link CurvatureRepricing} per currency curve.
 */
public final class GirrSbm {

  private GirrSbm() {
  }

  public static SbmCharge.Result compute(Sensitivities book, List<CurvatureRepricing> curvature) {
    return SbmCharge.of(GirrSbmParameters.baselDefault()).compute(book, curvature);
  }

  public static SbmCharge.Result compute(GirrSbmParameters params, Sensitivities book,
                                         List<CurvatureRepricing> curvature) {
    return SbmCharge.of(params).compute(book, curvature);
  }
}
