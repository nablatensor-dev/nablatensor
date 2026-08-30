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

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Splits a large replay into a series of {@link AadExecutable#replay} dispatches,
 * each sized from a measured rate to stay within a per-dispatch budget, and
 * combines the partial means (and their standard errors) into one result.
 *
 * <p>This backs the {@code default} chunking methods on {@link AadExecutable} so
 * that an engine only has to implement a single unbounded {@code replay}. Per
 * executable it keeps a learned throughput estimate, so repeated calls do not
 * re-probe. {@link AbstractAadExecutable} carries its own equivalent and never
 * touches this class.
 */
final class ChunkedReplay {

  private static final long PROBE_PATHS = 65_536L;
  private static final long MIN_CHUNK_PATHS = 4_096L;

  private static final Map<AadExecutable, ChunkedReplay> BY_EXECUTABLE = new ConcurrentHashMap<>();

  static ChunkedReplay of(AadExecutable executable) {
    return BY_EXECUTABLE.computeIfAbsent(executable, ChunkedReplay::new);
  }

  static void forget(AadExecutable executable) {
    BY_EXECUTABLE.remove(executable);
  }

  private final AadExecutable executable;
  private final double maxChunkSeconds;
  private double pathsPerSecond;

  private ChunkedReplay(AadExecutable executable) {
    this.executable = executable;
    String override = System.getProperty("nablatensor.maxLaunchSeconds");
    this.maxChunkSeconds = override != null ? Double.parseDouble(override) : 2.0;
  }

  double maxChunkSeconds() {
    return maxChunkSeconds;
  }

  double measuredPathsPerSecond() {
    return pathsPerSecond;
  }

  double calibrate(long seed) {
    long paths = PROBE_PATHS;
    for (int attempt = 0; attempt < 12; attempt++) {
      AadResult probe = executable.replay(paths, 0L, seed);
      observe(paths, probe.seconds());
      if (probe.seconds() >= maxChunkSeconds * 0.2) {
        break;
      }
      long next = budgetedChunk();
      if (next <= paths) {
        break;
      }
      paths = Math.min(next, paths * 8L);
    }
    return pathsPerSecond;
  }

  AadResult replaySafe(long totalPaths, long seed) {
    if (totalPaths <= 0) {
      throw new IllegalArgumentException("paths must be positive");
    }
    if (pathsPerSecond <= 0.0) {
      calibrate(seed);
    }
    AadResultAccumulator acc = null;
    long done = 0;
    while (done < totalPaths) {
      long chunk = Math.min(budgetedChunk(), totalPaths - done);
      AadResult partial = executable.replay(chunk, done, seed);
      if (acc == null) {
        acc = new AadResultAccumulator(partial);
      }
      acc.add(partial, chunk, totalPaths);
      observe(chunk, partial.seconds());
      done += chunk;
    }
    return acc.result(totalPaths);
  }

  private void observe(long paths, double seconds) {
    if (seconds <= 0.0) {
      return;
    }
    double sample = paths / seconds;
    pathsPerSecond = pathsPerSecond <= 0.0 ? sample : 0.5 * (pathsPerSecond + sample);
  }

  private long budgetedChunk() {
    if (pathsPerSecond <= 0.0) {
      return PROBE_PATHS;
    }
    return Math.max(MIN_CHUNK_PATHS, (long) (pathsPerSecond * maxChunkSeconds));
  }
}
