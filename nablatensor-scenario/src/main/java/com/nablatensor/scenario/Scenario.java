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
package com.nablatensor.scenario;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** A named bundle of {@link Shock}s applied together to a base market. */
public record Scenario(String name, List<Shock> shocks) {

  public Scenario {
    shocks = List.copyOf(shocks);
  }

  public static Scenario of(String name, Shock... shocks) {
    return new Scenario(name, List.of(shocks));
  }

  /** The base state, its shocks applied. Inputs not mentioned are left as they are. */
  public Map<String, Double> apply(Map<String, Double> base) {
    Map<String, Double> out = new LinkedHashMap<>(base);
    for (Shock s : shocks) {
      double b = out.getOrDefault(s.input(), Double.NaN);
      if (Double.isNaN(b) && s.kind() != Shock.Kind.ABSOLUTE) {
        throw new IllegalArgumentException(
            "scenario '" + name + "' shocks unknown input '" + s.input() + "' relatively/additively");
      }
      out.put(s.input(), s.shocked(Double.isNaN(b) ? 0.0 : b));
    }
    return out;
  }
}
