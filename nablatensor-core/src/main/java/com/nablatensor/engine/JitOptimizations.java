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

import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * Opt-in code-generation optimizations for the {@code cpu-jit} AAD engine.
 *
 * <p>The default is {@link Level#NONE}: the generated kernel is a straight-line
 * transcription of the recorded tape that calls {@code java.lang.Math} for every
 * transcendental, so its output matches the scalar {@code cpu} interpreter bit
 * for bit. Every optimization below is enabled <em>explicitly</em>, either by
 * {@linkplain #level(Level) picking a level} or by
 * {@linkplain #of(Category...) naming individual categories}.
 *
 * <pre>{@code
 * Nabla.model(base, this::asian)
 *     .greeks().fp64().threads(8).on("cpu-jit")
 *     .jit(JitOptimizations.Level.LOW_RISK)          // level form
 *     .build();
 *
 * Nabla.model(base, this::asian)
 *     .greeks().fp64().threads(8).on("cpu-jit")
 *     .jit(JitOptimizations.Category.ROLLED_LOOPS,   // category form
 *          JitOptimizations.Category.DRAW_CACHE)
 *     .build();
 * }</pre>
 *
 * <p>The system properties {@code -Dnablatensor.jit.roll} and
 * {@code -Dnablatensor.crn} still override this setting when present, so existing
 * benchmark scripts are unaffected; the fluent value is what applies when they
 * are absent.
 */
public final class JitOptimizations {

  /** A single, independently switchable {@code cpu-jit} optimization. */
  public enum Category {

    /**
     * Detect a {@code prologue · body×N · epilogue} tape (a Monte-Carlo path
     * loop) and emit the body once inside a bytecode {@code for} loop, with
     * per-iteration values held in JVM locals and a compact {@code v[]}/{@code
     * d[]} that covers only the live nodes. Collapses the generated class from
     * tens of KB to ~1.5 KB (removing a HotSpot {@code HugeMethodLimit}
     * fallback) and cuts the per-path arithmetic floor by roughly a quarter.
     *
     * <p>Bit-exact against the flat kernel at more than one worker thread;
     * single-threaded it can differ by about 0.1 ULP because C2 contracts the
     * loop's {@code local += x*y} to a fused multiply-add while the flat
     * kernel's {@code array[i] += x*y} stays two rounded operations — a
     * difference many orders of magnitude below Monte-Carlo error. Included in
     * {@link Level#LOW_RISK}.
     */
    ROLLED_LOOPS,

    /**
     * Common-random-numbers caching: the first replay of a given
     * {@code (seed, pathOffset, paths)} block generates the whole normal-draw
     * array once and keeps it; later replays of the same block — a revaluation
     * under a shocked market — reuse it and skip the RNG entirely. Bit-exact
     * (the draws do not depend on the market), and the largest single lever for
     * the risk-run revaluation loop, but it holds the draw block in memory
     * (bounded by {@code -Dnablatensor.crn.cap} elements) and does nothing for a
     * single non-repeated run. Enabled only by {@link Level#ALL} or by naming
     * this category.
     */
    DRAW_CACHE,

    /**
     * Replace {@code Math.exp}/{@code log}/{@code sin}/{@code cos} in the
     * Box-Muller draw generator with bounded-range polynomial approximations.
     * Changes the low bits of every path (prices then agree only to about four
     * significant figures, as with the SIMD engine) and, measured on Zen 4, is
     * slower than the SVML-backed {@code Math} intrinsics it replaces. Enabled
     * only by {@link Level#ALL} or by naming this category.
     */
    FAST_MATH
  }

  /** A named bundle of {@link Category} values. */
  public enum Level {

    /** No optimizations. The default; the kernel matches the scalar interpreter. */
    NONE,

    /**
     * Optimizations that stay within about 1 ULP of the unoptimized kernel and
     * carry no memory or accuracy trade-off: {@link Category#ROLLED_LOOPS}.
     */
    LOW_RISK,

    /**
     * Every optimization, including ones that cost memory
     * ({@link Category#DRAW_CACHE}) or trade accuracy and are not always faster
     * ({@link Category#FAST_MATH}).
     */
    ALL
  }

  /** The default: no optimizations. */
  public static final JitOptimizations NONE = new JitOptimizations(EnumSet.noneOf(Category.class));

  private final Set<Category> enabled;

  private JitOptimizations(Set<Category> enabled) {
    this.enabled = enabled;
  }

  /** The optimizations bundled under {@code level}. */
  public static JitOptimizations level(Level level) {
    return switch (Objects.requireNonNull(level, "level")) {
      case NONE -> NONE;
      case LOW_RISK -> of(Category.ROLLED_LOOPS);
      case ALL -> of(Category.values());
    };
  }

  /** Exactly the named categories, and nothing else. */
  public static JitOptimizations of(Category... categories) {
    EnumSet<Category> set = EnumSet.noneOf(Category.class);
    for (Category category : categories) {
      set.add(Objects.requireNonNull(category, "category"));
    }
    return set.isEmpty() ? NONE : new JitOptimizations(set);
  }

  /** This set plus {@code category}. */
  public JitOptimizations with(Category category) {
    EnumSet<Category> set = EnumSet.noneOf(Category.class);
    set.addAll(enabled);
    set.add(Objects.requireNonNull(category, "category"));
    return new JitOptimizations(set);
  }

  /** Whether {@code category} is enabled. */
  public boolean has(Category category) {
    return enabled.contains(category);
  }

  /** The enabled categories, unmodifiable. */
  public Set<Category> categories() {
    return Collections.unmodifiableSet(enabled);
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof JitOptimizations that && enabled.equals(that.enabled);
  }

  @Override
  public int hashCode() {
    return enabled.hashCode();
  }

  @Override
  public String toString() {
    return enabled.isEmpty() ? "jit[NONE]" : "jit" + enabled;
  }
}
