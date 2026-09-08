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

import java.util.ArrayList;
import java.util.List;

/** A collection of {@link Scenario}s: an explicit list, a 1-D ladder, or the cartesian grid of several ladders. */
public record ScenarioSet(List<Scenario> scenarios) {

  public ScenarioSet {
    scenarios = List.copyOf(scenarios);
  }

  public static ScenarioSet list(Scenario... scenarios) {
    return new ScenarioSet(List.of(scenarios));
  }

  public static ScenarioSet ladder(Ladder ladder) {
    return new ScenarioSet(ladder.scenarios());
  }

  /** Cartesian product of the ladders; each combined scenario merges one point from each. */
  public static ScenarioSet grid(Ladder... ladders) {
    List<Scenario> acc = new ArrayList<>();
    acc.add(Scenario.of(""));
    for (Ladder l : ladders) {
      List<Scenario> next = new ArrayList<>();
      for (Scenario base : acc) {
        for (Scenario point : l.scenarios()) {
          List<Shock> merged = new ArrayList<>(base.shocks());
          merged.addAll(point.shocks());
          String name = base.name().isEmpty() ? point.name() : base.name() + ", " + point.name();
          next.add(new Scenario(name, merged));
        }
      }
      acc = next;
    }
    return new ScenarioSet(acc);
  }

  public int size() {
    return scenarios.size();
  }
}
