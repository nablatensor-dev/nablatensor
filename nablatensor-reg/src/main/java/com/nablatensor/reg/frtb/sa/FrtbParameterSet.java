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
package com.nablatensor.reg.frtb.sa;

import com.nablatensor.reg.frtb.sbm.commodity.CommoditySbmParameters;
import com.nablatensor.reg.frtb.sbm.csr.CsrSbmParameters;
import com.nablatensor.reg.frtb.sbm.equity.EquitySbmProfile;
import com.nablatensor.reg.frtb.sbm.fx.FxSbmParameters;
import com.nablatensor.reg.frtb.sbm.girr.GirrSbmParameters;
import com.nablatensor.risk.RiskClass;
import com.nablatensor.risk.RiskClassProfile;
import java.util.EnumMap;
import java.util.Map;

/**
 * A named bundle of the seven {@link RiskClassProfile}s the {@link FrtbSa}
 * assembler runs. One implementation per rulebook variant — the Basel default is
 * {@link #baselMar21()}; EU / UK / EU-2027-relief variants (Phase 4) plug in
 * here without touching the assembler.
 */
public interface FrtbParameterSet {

  /** Human-readable name for reporting / the COREP dual. */
  String name();

  /** The profile for a risk class. */
  RiskClassProfile profile(RiskClass riskClass);

  /** The Basel Framework MAR21 parameter set (published values; CSR sec / CTP are placeholders). */
  static FrtbParameterSet baselMar21() {
    Map<RiskClass, RiskClassProfile> m = new EnumMap<>(RiskClass.class);
    m.put(RiskClass.GIRR, GirrSbmParameters.baselDefault());
    m.put(RiskClass.CSR_NON_SEC, CsrSbmParameters.nonSec());
    m.put(RiskClass.CSR_SEC, CsrSbmParameters.securitisation());
    m.put(RiskClass.CSR_SEC_CTP, CsrSbmParameters.ctp());
    m.put(RiskClass.EQUITY, EquitySbmProfile.INSTANCE);
    m.put(RiskClass.COMMODITY, CommoditySbmParameters.INSTANCE);
    m.put(RiskClass.FX, FxSbmParameters.INSTANCE);
    return of("Basel MAR21", m);
  }

  /** A parameter set from an explicit per-class map. */
  static FrtbParameterSet of(String name, Map<RiskClass, RiskClassProfile> profiles) {
    Map<RiskClass, RiskClassProfile> copy = new EnumMap<>(RiskClass.class);
    copy.putAll(profiles);
    return new FrtbParameterSet() {
      @Override
      public String name() {
        return name;
      }

      @Override
      public RiskClassProfile profile(RiskClass riskClass) {
        RiskClassProfile p = copy.get(riskClass);
        if (p == null) {
          throw new IllegalArgumentException("no profile for risk class " + riskClass + " in set '" + name + "'");
        }
        return p;
      }
    };
  }
}
