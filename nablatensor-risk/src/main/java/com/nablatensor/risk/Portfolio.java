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
package com.nablatensor.risk;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A book of trades. Each trade is recorded and risked on its own tape; the
 * portfolio and netting-set views are pure addition of the resulting
 * {@link Sensitivities} — the aggregation layer never touches a kernel (Seam 7).
 */
public record Portfolio(List<Trade> trades) {

  /** A trade: an id, its netting set, and its risk-factor sensitivities. */
  public interface Trade {
    String id();

    String nettingSet();

    Sensitivities sensitivities();
  }

  public Portfolio {
    trades = List.copyOf(trades);
  }

  public static Portfolio of(Trade... trades) {
    return new Portfolio(List.of(trades));
  }

  /** Book-level sensitivities: the sum over every trade. */
  public Sensitivities aggregate() {
    Sensitivities acc = Sensitivities.empty();
    for (Trade t : trades) {
      acc = acc.plus(t.sensitivities());
    }
    return acc;
  }

  /** Sensitivities per netting set, in first-seen order. */
  public Map<String, Sensitivities> byNettingSet() {
    Map<String, Sensitivities> out = new LinkedHashMap<>();
    for (Trade t : trades) {
      out.merge(t.nettingSet(), t.sensitivities(), Sensitivities::plus);
    }
    return out;
  }

  /** A plain immutable trade holding a pre-computed sensitivity vector. */
  public static Trade trade(String id, String nettingSet, Sensitivities sensitivities) {
    return new Trade() {
      @Override
      public String id() {
        return id;
      }

      @Override
      public String nettingSet() {
        return nettingSet;
      }

      @Override
      public Sensitivities sensitivities() {
        return sensitivities;
      }
    };
  }
}
