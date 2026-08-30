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

/**
 * A one-dimensional grid of {@link Scenario}s over a single input.
 *
 * <pre>{@code
 * Ladder spot = Ladder.of("S0").absolute().from(80).to(120).step(5);   // 80, 85, ... 120
 * Ladder vol  = Ladder.of("sigma").additive().from(-0.05).to(0.05).points(11);
 * }</pre>
 */
public final class Ladder {

  private final String input;
  private final Shock.Kind kind;
  private final double[] values;

  private Ladder(String input, Shock.Kind kind, double[] values) {
    this.input = input;
    this.kind = kind;
    this.values = values;
  }

  public static Builder of(String input) {
    return new Builder(input);
  }

  public String input() {
    return input;
  }

  public double[] values() {
    return values.clone();
  }

  public int size() {
    return values.length;
  }

  /** One {@link Scenario} per grid point, named {@code input=value}. */
  public List<Scenario> scenarios() {
    List<Scenario> out = new ArrayList<>(values.length);
    for (double v : values) {
      out.add(Scenario.of(input + "=" + trim(v), new Shock(input, kind, v)));
    }
    return out;
  }

  private static String trim(double v) {
    return v == Math.rint(v) ? Long.toString((long) v) : Double.toString(v);
  }

  /** Fluent construction of a {@link Ladder}. */
  public static final class Builder {
    private final String input;
    private Shock.Kind kind = Shock.Kind.ABSOLUTE;
    private double from;
    private double to;

    private Builder(String input) {
      this.input = input;
    }

    public Builder absolute() {
      this.kind = Shock.Kind.ABSOLUTE;
      return this;
    }

    public Builder relative() {
      this.kind = Shock.Kind.RELATIVE;
      return this;
    }

    public Builder additive() {
      this.kind = Shock.Kind.ADDITIVE;
      return this;
    }

    public Builder from(double v) {
      this.from = v;
      return this;
    }

    public Builder to(double v) {
      this.to = v;
      return this;
    }

    public Ladder step(double h) {
      if (!(h > 0) || to < from) {
        throw new IllegalArgumentException("need to >= from and step > 0");
      }
      int n = (int) Math.round((to - from) / h) + 1;
      return points(n);
    }

    public Ladder points(int n) {
      if (n < 1) {
        throw new IllegalArgumentException("need at least one point");
      }
      double[] v = new double[n];
      for (int i = 0; i < n; i++) {
        v[i] = n == 1 ? from : from + (to - from) * i / (n - 1);
      }
      return new Ladder(input, kind, v);
    }
  }
}
