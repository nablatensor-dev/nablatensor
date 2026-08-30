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
import com.nablatensor.engine.Nabla;
import java.lang.reflect.RecordComponent;
import java.util.function.BiConsumer;

/**
 * Shared helpers for the Phase-1 tests: build a recorded valuation over a record
 * market and read its adjoint gradient, or a central-bump gradient by component
 * index, all with common random numbers so the two agree to the bump's own
 * {@code O(h^2)} error rather than to Monte-Carlo noise.
 */
final class Phase1Support {

  static final long SCENARIOS = 200_000L;
  static final long SEED = 20260830L;

  private Phase1Support() {
  }

  /** Component names of a record market, in declaration order. */
  static String[] names(Class<? extends Record> market) {
    RecordComponent[] rc = market.getRecordComponents();
    String[] n = new String[rc.length];
    for (int i = 0; i < rc.length; i++) {
      n[i] = rc[i].getName();
    }
    return n;
  }

  /** {@code [price, dValue/dComp_0, dValue/dComp_1, ...]} from one build + one adjoint replay. */
  static <M extends Record> double[] adjoint(M market, BiConsumer<AadRecorder, Nabla.Inputs<M>> valuation) {
    RecordComponent[] rc = market.getClass().getRecordComponents();
    try (Nabla.TypedPricer<M> pricer = Nabla.model(market, valuation).fp64().greeks().on("cpu-jit").build()) {
      Nabla.TypedValuation<M> v = pricer.value().with(market).scenarios(SCENARIOS).seed(SEED).run();
      double[] out = new double[rc.length + 1];
      out[0] = v.price();
      // the gradient comes back as a record of the same shape; read it component-by-component
      Object greeks = v.greeks();
      for (int i = 0; i < rc.length; i++) {
        try {
          out[i + 1] = ((Number) rc[i].getAccessor().invoke(greeks)).doubleValue();
        } catch (ReflectiveOperationException e) {
          throw new RuntimeException(e);
        }
      }
      return out;
    }
  }

  /** Central-bump {@code dValue/dComp_index} on the same seed. */
  static <M extends Record> double bump(M market, BiConsumer<AadRecorder, Nabla.Inputs<M>> valuation,
                                        int index, double h) {
    return (priceAt(shift(market, index, h), valuation) - priceAt(shift(market, index, -h), valuation)) / (2 * h);
  }

  static <M extends Record> double priceAt(M market, BiConsumer<AadRecorder, Nabla.Inputs<M>> valuation) {
    try (Nabla.TypedPricer<M> pricer = Nabla.model(market, valuation).fp64().priceOnly().on("cpu-jit").build()) {
      return pricer.value().with(market).scenarios(SCENARIOS).seed(SEED).run().price();
    }
  }

  @SuppressWarnings("unchecked")
  static <M extends Record> M shift(M market, int index, double delta) {
    try {
      RecordComponent[] rc = market.getClass().getRecordComponents();
      Object[] args = new Object[rc.length];
      Class<?>[] types = new Class<?>[rc.length];
      for (int i = 0; i < rc.length; i++) {
        types[i] = rc[i].getType();
        double val = ((Number) rc[i].getAccessor().invoke(market)).doubleValue();
        args[i] = i == index ? val + delta : val;
      }
      return (M) market.getClass().getDeclaredConstructor(types).newInstance(args);
    } catch (ReflectiveOperationException e) {
      throw new RuntimeException(e);
    }
  }
}
