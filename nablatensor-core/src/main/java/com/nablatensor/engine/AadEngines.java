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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.ServiceLoader;

/**
 * Discovers {@link AadEngine} implementations on the module path and picks one.
 *
 * <p>Selection is by descending priority among engines that are both available
 * and able to honour the options, so an accelerator wins whenever there is one.
 * A specific engine can be demanded with {@code -Dnablatensor.engine=simd} or by
 * calling {@link #require(String)}, which is the intended way to benchmark a
 * backend that would otherwise never be chosen.
 */
public final class AadEngines {

  public static final String ENGINE_PROPERTY = "nablatensor.engine";

  private AadEngines() {
  }

  /** Every engine on the module path, including unavailable ones. */
  public static List<AadEngine> discovered() {
    List<AadEngine> engines = new ArrayList<>();
    // An engine whose optional runtime dependency is missing must not take the
    // whole enumeration down with it, so provider construction is isolated.
    ServiceLoader<AadEngine> loader = ServiceLoader.load(AadEngine.class);
    for (var iterator = loader.stream().iterator(); iterator.hasNext(); ) {
      try {
        engines.add(iterator.next().get());
      } catch (Throwable ignored) {
        // Provider could not be constructed here; treat it as absent.
      }
    }
    engines.sort(Comparator.comparingInt(AadEngine::priority).reversed());
    return engines;
  }

  public static List<AadEngine> available(AadOptions options) {
    List<AadEngine> usable = new ArrayList<>();
    for (AadEngine engine : discovered()) {
      if (isUsable(engine, options)) {
        usable.add(engine);
      }
    }
    return usable;
  }

  public static Optional<AadEngine> find(String name, AadOptions options) {
    for (AadEngine engine : discovered()) {
      if (engine.name().equalsIgnoreCase(name) && isUsable(engine, options)) {
        return Optional.of(engine);
      }
    }
    return Optional.empty();
  }

  /** The named engine, or a failure explaining what was found instead. */
  public static AadEngine require(String name, AadOptions options) {
    return find(name, options).orElseThrow(() -> new IllegalStateException(
        "AAD engine '" + name + "' is not usable here; available: " + names(available(options))));
  }

  /** Highest-priority usable engine, honouring the engine system property. */
  public static AadEngine select(AadOptions options) {
    String requested = System.getProperty(ENGINE_PROPERTY);
    if (requested != null && !requested.isBlank()) {
      return require(requested.trim(), options);
    }
    List<AadEngine> usable = available(options);
    if (usable.isEmpty()) {
      throw new IllegalStateException(
          "no usable AAD engine; discovered: " + names(discovered()));
    }
    return usable.get(0);
  }

  public static AadExecutable compile(AadTape tape, AadOptions options) {
    return select(options).compile(tape, options);
  }

  private static boolean isUsable(AadEngine engine, AadOptions options) {
    try {
      return engine.isAvailable() && engine.supports(options);
    } catch (Throwable ignored) {
      return false;
    }
  }

  private static String names(List<AadEngine> engines) {
    return engines.isEmpty() ? "(none)" : String.join(", ", engines.stream().map(AadEngine::name).toList());
  }
}
