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

import com.nablatensor.codegen.Of;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Replay kernel settings.
 *
 * @param precision working precision inside the kernel; consumer/workstation
 *     NVIDIA parts run FP64 at a small fraction of FP32 throughput, so FLOAT32
 *     is the default and FLOAT64 exists for accuracy checks
 * @param adjoints whether to emit the reverse sweep; a value-only kernel is
 *     what a price-without-Greeks run costs
 * @param threads worker threads for CPU-side engines; {@code 0} means one per
 *     available processor. Ignored by the CUDA engine.
 * @param jit opt-in code-generation optimizations for the {@code cpu-jit}
 *     engine; {@link JitOptimizations#NONE} by default, and ignored by every
 *     other engine
 * @param engineOptions free-form per-engine tuning, keyed by engine name then by
 *     option key; an engine reads only its own namespace and ignores unknown
 *     keys. {@link JitOptimizations} stays the typed, canonical surface for
 *     {@code cpu-jit}; this is the generic mechanism for the rest.
 */
@Of
public final class AadOptions {

  private final PrecisionEnum precision;
  private final boolean adjoints;
  private final int threads;
  private final JitOptimizations jit;
  private final Map<String, Map<String, String>> engineOptions;
  static AadOptions create(PrecisionEnum precision, boolean adjoints, int threads, JitOptimizations jit, Map<String, Map<String, String>> engineOptions) {
    return new AadOptions(precision, adjoints, threads, jit, engineOptions);
  }

  public static AadOptionsBuilder of() {
    return new AadOptionsBuilder();
  }

  public PrecisionEnum precision() {
    return precision;
  }

  public boolean adjoints() {
    return adjoints;
  }

  public int threads() {
    return threads;
  }

  public JitOptimizations jit() {
    return jit;
  }

  public Map<String, Map<String, String>> engineOptions() {
    return engineOptions;
  }


  public enum PrecisionEnum {
    /** Single precision — the GPU replay path (e.g. Vulkan) computes in fp32. */
    FLOAT32,
    /** Double precision — the CPU / SIMD default. */
    FLOAT64
  }

  private AadOptions(PrecisionEnum precision, boolean adjoints, int threads, JitOptimizations jit, Map<String, Map<String, String>> engineOptions) {
    if (jit == null) {
      jit = JitOptimizations.NONE;
    }
    engineOptions = deepImmutable(engineOptions);
    this.precision = precision;
    this.adjoints = adjoints;
    this.threads = threads;
    this.jit = jit;
    this.engineOptions = engineOptions;
  }

  private AadOptions(PrecisionEnum precision, boolean adjoints, int threads, JitOptimizations jit) {
    this(precision, adjoints, threads, jit, Map.of());
  }

  private AadOptions(PrecisionEnum precision, boolean adjoints, int threads) {
    this(precision, adjoints, threads, JitOptimizations.NONE, Map.of());
  }

  private AadOptions(PrecisionEnum precision, boolean adjoints) {
    this(precision, adjoints, 0, JitOptimizations.NONE, Map.of());
  }

  public static AadOptions defaults() {
    return AadOptions.of().precision(PrecisionEnum.FLOAT32).adjoints(true).threads(0).jit(JitOptimizations.NONE).engineOptions(Map.of()).build();
  }

  public AadOptions withPrecision(PrecisionEnum value) {
    return AadOptions.of().precision(value).adjoints(adjoints).threads(threads).jit(jit).engineOptions(engineOptions).build();
  }

  public AadOptions withAdjoints(boolean value) {
    return AadOptions.of().precision(precision).adjoints(value).threads(threads).jit(jit).engineOptions(engineOptions).build();
  }

  public AadOptions withThreads(int value) {
    return AadOptions.of().precision(precision).adjoints(adjoints).threads(value).jit(jit).engineOptions(engineOptions).build();
  }

  public AadOptions withJit(JitOptimizations value) {
    return AadOptions.of().precision(precision).adjoints(adjoints).threads(threads).jit(value).engineOptions(engineOptions).build();
  }

  /** This, plus one tuning option for the named engine. */
  public AadOptions withEngineOption(String engine, String key, String value) {
    Map<String, Map<String, String>> copy = new LinkedHashMap<>();
    engineOptions.forEach((e, m) -> copy.put(e, new LinkedHashMap<>(m)));
    copy.computeIfAbsent(engine, e -> new LinkedHashMap<>()).put(key, value);
    return AadOptions.of().precision(precision).adjoints(adjoints).threads(threads).jit(jit).engineOptions(copy).build();
  }

  /** The tuning options for one engine; never {@code null}. */
  public Map<String, String> engineOptions(String engine) {
    return engineOptions.getOrDefault(engine, Map.of());
  }

  /** Resolved worker count, with {@code 0} meaning one per available processor. */
  public int resolvedThreads() {
    return threads > 0 ? threads : Runtime.getRuntime().availableProcessors();
  }

  private static Map<String, Map<String, String>> deepImmutable(Map<String, Map<String, String>> in) {
    if (in == null || in.isEmpty()) {
      return Map.of();
    }
    Map<String, Map<String, String>> copy = new LinkedHashMap<>();
    in.forEach((e, m) -> copy.put(e, Map.copyOf(m)));
    return Collections.unmodifiableMap(copy);
  }
}
