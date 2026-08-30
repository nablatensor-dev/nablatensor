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
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.function.Predicate;

/**
 * An immutable {@code RiskFactor -> sensitivity} vector — the output of one
 * adjoint sweep, mapped onto regulatory risk factors. Composition (portfolio and
 * netting-set aggregation) is plain addition here, entirely outside the tape.
 */
public final class Sensitivities {

  private final Map<RiskFactor, Double> values;

  private Sensitivities(Map<RiskFactor, Double> values) {
    this.values = Map.copyOf(values);
  }

  public static Sensitivities empty() {
    return new Sensitivities(Map.of());
  }

  public static Builder builder() {
    return new Builder();
  }

  public double get(RiskFactor factor) {
    return values.getOrDefault(factor, 0.0);
  }

  public Map<RiskFactor, Double> asMap() {
    return values;
  }

  public boolean isEmpty() {
    return values.isEmpty();
  }

  /** Element-wise sum. */
  public Sensitivities plus(Sensitivities other) {
    Map<RiskFactor, Double> merged = new LinkedHashMap<>(values);
    other.values.forEach((k, v) -> merged.merge(k, v, Double::sum));
    return new Sensitivities(merged);
  }

  public Sensitivities scaled(double factor) {
    Map<RiskFactor, Double> out = new LinkedHashMap<>();
    values.forEach((k, v) -> out.put(k, v * factor));
    return new Sensitivities(out);
  }

  public Sensitivities filter(Predicate<RiskFactor> keep) {
    Map<RiskFactor, Double> out = new LinkedHashMap<>();
    values.forEach((k, v) -> {
      if (keep.test(k)) {
        out.put(k, v);
      }
    });
    return new Sensitivities(out);
  }

  public Sensitivities ofClass(RiskClass rc) {
    return filter(f -> f.riskClass() == rc);
  }

  public Sensitivities ofMeasure(RiskMeasure m) {
    return filter(f -> f.measure() == m);
  }

  /** The distinct buckets present, in natural order. */
  public SortedSet<String> buckets() {
    SortedSet<String> b = new TreeSet<>();
    values.keySet().forEach(f -> b.add(f.bucket()));
    return b;
  }

  public Sensitivities inBucket(String bucket) {
    return filter(f -> f.bucket().equals(bucket));
  }

  public static final class Builder {
    private final Map<RiskFactor, Double> v = new LinkedHashMap<>();

    public Builder add(RiskFactor factor, double sensitivity) {
      v.merge(factor, sensitivity, Double::sum);
      return this;
    }

    public Sensitivities build() {
      return new Sensitivities(v);
    }
  }
}
