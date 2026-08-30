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
package com.nablatensor.engine.cpu;

/**
 * Philox2x32-10 keyed on the scenario index, mirroring the generator emitted
 * into the CUDA kernel so the two line up path for path.
 *
 * <p>One instance is a single random stream. Stream {@code 0} is byte-for-byte
 * the historical single-stream generator; a named stream folds its index into
 * the key so its draws are independent of every other stream's. {@link #normal()}
 * yields a Box-Muller pair per Philox counter (cosine leg for an even draw
 * ordinal, sine leg for the odd one); {@link #uniform()} takes its own,
 * high-tagged counter range so uniforms never alias normals.
 */
public final class Philox {

  private static final double UINT_SCALE = 2.3283064365386963e-10;
  private static final int UNIFORM_TAG = 0x40000000;

  private final int lo;
  private final int hi;
  private int counter;
  private int uniformCounter;
  private double spare;
  private boolean has;

  public Philox(long path, long seed) {
    this(path, seed, 0);
  }

  public Philox(long path, long seed, int stream) {
    int l = (int) path ^ (int) seed;
    int h = (int) (path >>> 32) ^ (int) (seed >>> 32);
    if (stream != 0) {
      h ^= stream * 0x9E3779B1;   // independent per stream; stream 0 unchanged
    }
    this.lo = l;
    this.hi = h;
  }

  public double normal() {
    if (has) {
      has = false;
      return spare;
    }
    int c0 = lo;
    int c1 = hi ^ counter++;
    int key = 0x1BD11BDA;
    for (int i = 0; i < 10; i++) {
      long product = 0xD256D193L * Integer.toUnsignedLong(c0);
      int high = (int) (product >>> 32);
      int low = (int) product;
      c0 = high ^ key ^ c1;
      c1 = low;
      key += 0x9E3779B9;
    }
    double u1 = (Integer.toUnsignedLong(c0) + 0.5) * UINT_SCALE;
    double u2 = (Integer.toUnsignedLong(c1) + 0.5) * UINT_SCALE;
    double radius = Math.sqrt(-2.0 * Math.log(u1));
    double angle = 6.283185307179586 * u2;
    spare = radius * Math.sin(angle);
    has = true;
    return radius * Math.cos(angle);
  }

  /** A uniform draw on {@code [0, 1)}. */
  public double uniform() {
    int c0 = lo;
    int c1 = hi ^ (uniformCounter++ | UNIFORM_TAG);
    int key = 0x1BD11BDA;
    for (int i = 0; i < 10; i++) {
      long product = 0xD256D193L * Integer.toUnsignedLong(c0);
      int high = (int) (product >>> 32);
      int low = (int) product;
      c0 = high ^ key ^ c1;
      c1 = low;
      key += 0x9E3779B9;
    }
    return (Integer.toUnsignedLong(c0) + 0.5) * UINT_SCALE;
  }
}
