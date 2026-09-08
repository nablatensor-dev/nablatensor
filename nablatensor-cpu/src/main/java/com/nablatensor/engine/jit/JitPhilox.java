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
package com.nablatensor.engine.jit;

/**
 * Per-path random generation for the JIT engine. The default stream's normal
 * draws are bit-identical to {@code com.nablatensor.engine.cpu.Philox} so the
 * generated kernel's prices match the scalar reference; a named stream folds its
 * index into the key, and uniforms take a high-tagged counter range so they
 * never alias normals. {@link JitReplay} fills the flat draw buffer once per
 * path before invoking the kernel, which then just indexes it.
 */
public final class JitPhilox {

  private static final double UINT_SCALE = 2.3283064365386963e-10;
  private static final double TWO_PI = 6.283185307179586;
  private static final int UNIFORM_TAG = 0x40000000;

  private static final boolean FAST_TRIG_PROP = "fast".equals(System.getProperty("nablatensor.jit.trig"));
  private static final boolean FAST_LOG_PROP = "fast".equals(System.getProperty("nablatensor.jit.log"));

  private JitPhilox() {
  }

  /** Default-stream normals into {@code out[0 .. count)}, fast math off (legacy signature). */
  public static void fillNormals(double[] out, int count, long path, long seed) {
    fillNormals(out, 0, count, path, seed, 0, false);
  }

  /** Default-stream normals into {@code out[0 .. count)} (legacy signature). */
  public static void fillNormals(double[] out, int count, long path, long seed, boolean fastMath) {
    fillNormals(out, 0, count, path, seed, 0, fastMath);
  }

  /**
   * Fills {@code out[off .. off+count)} with stream {@code stream}'s standard-normal
   * draws for this path. Each Philox block yields a Box-Muller pair (cosine leg
   * for an even index, sine leg for the odd one), so the transcendentals run
   * once per two draws. For {@code stream == 0} the stream is bit-for-bit the
   * historical single-stream generator.
   */
  public static void fillNormals(double[] out, int off, int count, long path, long seed,
                                 int stream, boolean fastMath) {
    final boolean fastTrig = fastMath || FAST_TRIG_PROP;
    final boolean fastLog = fastMath || FAST_LOG_PROP;
    final int seedLo = (int) path ^ (int) seed;
    final int seedHi = key(stream, (int) (path >>> 32) ^ (int) (seed >>> 32));
    final int pairs = (count + 1) >>> 1;
    for (int c = 0; c < pairs; c++) {
      int x0 = seedLo;
      int x1 = seedHi ^ c;
      int key = 0x1BD11BDA;
      for (int r = 0; r < 10; r++) {
        long product = 0xD256D193L * Integer.toUnsignedLong(x0);
        int hi = (int) (product >>> 32);
        int lo = (int) product;
        x0 = hi ^ key ^ x1;
        x1 = lo;
        key += 0x9E3779B9;
      }
      double u1 = (Integer.toUnsignedLong(x0) + 0.5) * UINT_SCALE;
      double u2 = (Integer.toUnsignedLong(x1) + 0.5) * UINT_SCALE;
      double radius = Math.sqrt(-2.0 * (fastLog ? JitFastMath.log(u1) : Math.log(u1)));
      double angle = TWO_PI * u2;
      int j = c << 1;
      out[off + j] = radius * (fastTrig ? JitFastMath.cos(angle) : Math.cos(angle));
      if (j + 1 < count) {
        out[off + j + 1] = radius * (fastTrig ? JitFastMath.sin(angle) : Math.sin(angle));
      }
    }
  }

  /** Fills {@code out[off .. off+count)} with stream {@code stream}'s uniform {@code [0,1)} draws. */
  public static void fillUniforms(double[] out, int off, int count, long path, long seed, int stream) {
    final int seedLo = (int) path ^ (int) seed;
    final int seedHi = key(stream, (int) (path >>> 32) ^ (int) (seed >>> 32));
    for (int k = 0; k < count; k++) {
      int x0 = seedLo;
      int x1 = seedHi ^ (k | UNIFORM_TAG);
      int key = 0x1BD11BDA;
      for (int r = 0; r < 10; r++) {
        long product = 0xD256D193L * Integer.toUnsignedLong(x0);
        int hi = (int) (product >>> 32);
        int lo = (int) product;
        x0 = hi ^ key ^ x1;
        x1 = lo;
        key += 0x9E3779B9;
      }
      out[off + k] = (Integer.toUnsignedLong(x0) + 0.5) * UINT_SCALE;
    }
  }

  private static int key(int stream, int seedHi) {
    return stream == 0 ? seedHi : seedHi ^ (stream * 0x9E3779B1);
  }
}
