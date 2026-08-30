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

/**
 * A tape that has been compiled by some {@link AadEngine} and can be replayed.
 *
 * <p>Recording and compilation happen once; after that a replay costs one
 * dispatch to whatever backend produced this handle, no matter how many
 * scenarios it covers. Inputs are arguments rather than baked-in constants, so
 * {@link #setInput} re-prices a shifted market with no re-record and no
 * re-compile.
 *
 * <p>An engine implements the five core methods — {@link #engineName()},
 * {@link #tape()}, {@link #setInput}, {@link #replay} and {@link #close()}. The
 * segmentation methods ({@link #replaySafe}, {@link #calibrate},
 * {@link #measuredPathsPerSecond}, {@link #maxChunkSeconds}) and
 * {@link #compileSeconds} have working defaults that split a large replay into
 * budgeted dispatches; {@link AbstractAadExecutable} is an optional base that
 * provides the same with its own bookkeeping plus tape flattening.
 */
public interface AadExecutable extends AutoCloseable {

  /** Identifier of the engine that produced this handle, e.g. {@code cuda}. */
  String engineName();

  AadTape tape();

  void setInput(String name, double value);

  /** One dispatch covering {@code paths} scenarios starting at {@code pathOffset}. */
  AadResult replay(long paths, long pathOffset, long seed);

  @Override
  void close();

  /** Wall-clock compilation time, paid once. Zero for engines that interpret. */
  default double compileSeconds() {
    return 0.0;
  }

  /**
   * Replays {@code totalPaths} as a series of dispatches, each sized from a
   * measured rate to stay within {@link #maxChunkSeconds()}.
   */
  default AadResult replaySafe(long totalPaths, long seed) {
    return ChunkedReplay.of(this).replaySafe(totalPaths, seed);
  }

  /** Measures throughput with dispatches that start small and grow. */
  default double calibrate(long seed) {
    return ChunkedReplay.of(this).calibrate(seed);
  }

  default double measuredPathsPerSecond() {
    return ChunkedReplay.of(this).measuredPathsPerSecond();
  }

  /** Per-dispatch wall-clock budget used to size chunks. */
  default double maxChunkSeconds() {
    return ChunkedReplay.of(this).maxChunkSeconds();
  }
}
