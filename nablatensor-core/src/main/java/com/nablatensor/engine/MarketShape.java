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
package com.nablatensor.engine;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.RecordComponent;
import java.util.function.ToDoubleFunction;

/**
 * Reflective view of a market record: a point in risk-factor space, whose
 * gradient is a point of the same shape.
 *
 * <p>Component names come from the record, so the caller never writes one as a
 * string. An accessor method reference is resolved to a component index by
 * applying it to a probe instance whose components hold their own index —
 * which needs no bytecode inspection and no proxy.
 */
final class MarketShape<M extends Record> {

  // The probe holds index + PROBE_OFFSET rather than index, so that arithmetic on
  // an accessor (m -> m.spot() * 2) cannot land on a valid index and be mistaken
  // for one - which index 0 otherwise would, being a fixed point of most of it.
  private static final double PROBE_OFFSET = 0.25;

  private final Class<?> type;
  private final String[] names;
  private final Method[] accessors;
  private final Constructor<?> canonical;
  private final M probe;

  private MarketShape(Class<?> type, String[] names, Method[] accessors,
                      Constructor<?> canonical, M probe) {
    this.type = type;
    this.names = names;
    this.accessors = accessors;
    this.canonical = canonical;
    this.probe = probe;
  }

  static <M extends Record> MarketShape<M> of(M sample) {
    Class<?> type = sample.getClass();
    RecordComponent[] components = type.getRecordComponents();
    if (components == null || components.length == 0) {
      throw new IllegalArgumentException(type.getSimpleName() + " has no components");
    }
    String[] names = new String[components.length];
    Method[] accessors = new Method[components.length];
    Class<?>[] paramTypes = new Class<?>[components.length];
    for (int i = 0; i < components.length; i++) {
      if (components[i].getType() != double.class) {
        throw new IllegalArgumentException(type.getSimpleName() + "." + components[i].getName()
            + " is " + components[i].getType().getSimpleName()
            + "; every component of a market record must be a double");
      }
      names[i] = components[i].getName();
      accessors[i] = components[i].getAccessor();
      accessors[i].setAccessible(true);
      paramTypes[i] = double.class;
    }
    try {
      Constructor<?> canonical = type.getDeclaredConstructor(paramTypes);
      canonical.setAccessible(true);
      Object[] indices = new Object[components.length];
      for (int i = 0; i < components.length; i++) {
        indices[i] = i + PROBE_OFFSET;
      }
      @SuppressWarnings("unchecked")
      M probe = (M) canonical.newInstance(indices);
      return new MarketShape<>(type, names, accessors, canonical, probe);
    } catch (ReflectiveOperationException e) {
      throw new IllegalArgumentException("cannot use " + type.getSimpleName()
          + " as a market record", e);
    }
  }

  int size() {
    return names.length;
  }

  String name(int index) {
    return names[index];
  }

  double value(M market, int index) {
    try {
      return (double) accessors[index].invoke(market);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("cannot read " + names[index], e);
    }
  }

  /** Resolves a component accessor to its index by reading it off the probe. */
  int indexOf(ToDoubleFunction<M> accessor) {
    double probed = accessor.applyAsDouble(probe) - PROBE_OFFSET;
    int index = (int) probed;
    if (index != probed || index < 0 || index >= names.length) {
      throw new IllegalArgumentException(
          "expected a plain component accessor of " + type.getSimpleName()
              + ", such as " + type.getSimpleName() + "::" + names[0]);
    }
    return index;
  }

  M build(double[] values) {
    Object[] boxed = new Object[values.length];
    for (int i = 0; i < values.length; i++) {
      boxed[i] = values[i];
    }
    try {
      @SuppressWarnings("unchecked")
      M built = (M) canonical.newInstance(boxed);
      return built;
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("cannot build " + type.getSimpleName(), e);
    }
  }
}
