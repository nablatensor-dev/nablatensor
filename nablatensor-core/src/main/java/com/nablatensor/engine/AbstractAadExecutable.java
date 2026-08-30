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

import com.nablatensor.annotation.Internal;

/**
 * Chunking, calibration and rate tracking shared by every engine, on top of a
 * single abstract dispatch.
 *
 * <p>The reason chunking is in the base rather than in the CUDA engine alone is
 * that no backend benefits from one unbounded dispatch: a GPU that also drives a
 * display will have the driver kill an overlong kernel, and a CPU engine that
 * runs for minutes without returning cannot report progress or be interrupted.
 *
 * <p>Public only so the backend modules can extend it; not part of the supported
 * API. Implement {@link AadEngine} / {@link AadExecutable} directly instead.
 */
@Internal
public abstract class AbstractAadExecutable implements AadExecutable {

  private static final long PROBE_PATHS = 65_536L;
  private static final long MIN_CHUNK_PATHS = 4_096L;

  protected final AadTape tape;
  protected final AadOptions options;
  protected final double[] inputs;

  private double pathsPerSecond;
  private double maxChunkSeconds;
  private boolean closed;

  protected AbstractAadExecutable(AadTape tape, AadOptions options) {
    this.tape = tape;
    this.options = options;
    this.inputs = tape.recordedInputs();
    this.maxChunkSeconds = defaultMaxChunkSeconds();
  }

  /** Engines that must bound dispatch length more tightly override this. */
  protected double defaultMaxChunkSeconds() {
    String override = System.getProperty("nablatensor.maxLaunchSeconds");
    return override != null ? Double.parseDouble(override) : 2.0;
  }

  @Override
  public AadTape tape() {
    return tape;
  }

  @Override
  public double compileSeconds() {
    return 0.0;
  }

  @Override
  public double maxChunkSeconds() {
    return maxChunkSeconds;
  }

  public void setMaxChunkSeconds(double seconds) {
    if (!(seconds > 0.0)) {
      throw new IllegalArgumentException("maxChunkSeconds must be positive");
    }
    this.maxChunkSeconds = seconds;
  }

  @Override
  public void setInput(String name, double value) {
    int index = tape.inputNames().indexOf(name);
    if (index < 0) {
      throw new IllegalArgumentException("unknown input: " + name);
    }
    inputs[index] = value;
  }

  protected void checkOpen() {
    if (closed) {
      throw new IllegalStateException("executable is closed");
    }
  }

  @Override
  public final AadResult replaySafe(long totalPaths, long seed) {
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
      AadResult partial = replay(chunk, done, seed);
      if (acc == null) {
        acc = new AadResultAccumulator(partial);
      }
      acc.add(partial, chunk, totalPaths);
      observe(chunk, partial.seconds());
      done += chunk;
    }
    return acc.result(totalPaths);
  }

  @Override
  public final double calibrate(long seed) {
    long paths = PROBE_PATHS;
    for (int attempt = 0; attempt < 12; attempt++) {
      AadResult probe = replay(paths, 0L, seed);
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

  @Override
  public final double measuredPathsPerSecond() {
    return pathsPerSecond;
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

  @Override
  public void close() {
    closed = true;
  }
}
