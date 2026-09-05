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
package com.nablatensor.examples;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.nablatensor.tensor.Backend;
import com.nablatensor.tensor.BackendRegistry;
import com.nablatensor.tensor.ConvSpec;
import com.nablatensor.tensor.DType;
import com.nablatensor.tensor.Device;
import com.nablatensor.tensor.Op;
import com.nablatensor.tensor.Shape;
import com.nablatensor.tensor.spi.ComputeBackend;
import com.nablatensor.tensor.spi.DeviceBuffer;
import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Checks the Vulkan tensor backend against a plain-Java reference for the ops the
 * {@link ExamplesSmokeTest#mnistMlp} run does not exercise — transpose, the
 * two-phase whole-tensor reductions, axis reductions / argmax / their backward,
 * {@code sumAxis0}, rank-3 broadcast, and 2-D convolution with its gradients —
 * plus matmul on shapes that are not multiples of the block tile. Skips when no
 * Vulkan device is present.
 */
class VulkanBackendParityTest {

  private static ComputeBackend vk;

  @BeforeAll
  static void pick() {
    ComputeBackend backend = null;
    try {
      backend = BackendRegistry.forSelector(Backend.VULKAN);
    } catch (RuntimeException noVulkan) {
      // left null -> tests skip
    }
    assumeTrue(backend != null && backend.isAvailable(), "no Vulkan backend on this machine");
    vk = backend;
  }

  private static final Device DEV = Device.vulkan();

  private static float[] up(ComputeBackend b, float[] data, Shape shape) {
    return b.download(b.upload(data, shape, DType.F32, DEV));
  }

  private static float[] random(int n, long seed) {
    Random r = new Random(seed);
    float[] a = new float[n];
    for (int i = 0; i < n; i++) {
      a[i] = (float) (r.nextGaussian());
    }
    return a;
  }

  @Test
  void matmulOnAwkwardShapes() {
    // none of M, K, N is a multiple of the 64x64 block tile or the 16 K-tile
    for (int[] mkn : new int[][] {{37, 53, 41}, {1, 200, 1}, {64, 64, 64}, {130, 17, 129}}) {
      int m = mkn[0], k = mkn[1], n = mkn[2];
      float[] a = random(m * k, 1 + m);
      float[] b = random(k * n, 2 + n);
      float[] ref = new float[m * n];
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          double acc = 0;
          for (int p = 0; p < k; p++) {
            acc += (double) a[i * k + p] * b[p * n + j];
          }
          ref[i * n + j] = (float) acc;
        }
      }
      DeviceBuffer da = vk.upload(a, Shape.of(m, k), DType.F32, DEV);
      DeviceBuffer db = vk.upload(b, Shape.of(k, n), DType.F32, DEV);
      float[] got = vk.download(vk.matmul(da, db));
      assertArrayEquals(ref, got, 1e-3f, "matmul " + m + "x" + k + "x" + n);
    }
  }

  @Test
  void transpose2d() {
    int rows = 45, cols = 28;
    float[] a = random(rows * cols, 7);
    float[] ref = new float[rows * cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        ref[j * rows + i] = a[i * cols + j];
      }
    }
    float[] got = vk.download(vk.transpose(vk.upload(a, Shape.of(rows, cols), DType.F32, DEV)));
    assertArrayEquals(ref, got, 0f);
  }

  @Test
  void twoPhaseReductions() {
    int n = 300_000;                       // well past one 256-thread workgroup
    float[] a = random(n, 11);
    double sum = 0;
    float max = Float.NEGATIVE_INFINITY;
    for (float v : a) {
      sum += v;
      max = Math.max(max, v);
    }
    DeviceBuffer buf = vk.upload(a, Shape.of(n), DType.F32, DEV);
    assertArrayEquals(new float[] {(float) sum},
        vk.download(vk.reduceSum(buf)), Math.abs((float) sum) * 1e-4f + 1e-3f, "reduceSum");
    assertArrayEquals(new float[] {max}, vk.download(vk.reduceMax(buf)), 0f, "reduceMax");
  }

  @Test
  void axisReductionsAndArgmax() {
    int outer = 6, axis = 5, inner = 7;    // reduce the middle axis of (6,5,7)
    float[] a = random(outer * axis * inner, 21);
    Shape shape = Shape.of(outer, axis, inner);
    float[] sumRef = new float[outer * inner];
    float[] maxRef = new float[outer * inner];
    float[] argRef = new float[outer * inner];
    for (int o = 0; o < outer; o++) {
      for (int in = 0; in < inner; in++) {
        double s = 0;
        float best = Float.NEGATIVE_INFINITY;
        int bestIdx = 0;
        for (int x = 0; x < axis; x++) {
          float v = a[(o * axis + x) * inner + in];
          s += v;
          if (v > best) {
            best = v;
            bestIdx = x;
          }
        }
        sumRef[o * inner + in] = (float) s;
        maxRef[o * inner + in] = best;
        argRef[o * inner + in] = bestIdx;
      }
    }
    DeviceBuffer buf = vk.upload(a, shape, DType.F32, DEV);
    assertArrayEquals(sumRef, vk.download(vk.sumAxis(buf, 1, false)), 1e-4f, "sumAxis");
    assertArrayEquals(maxRef, vk.download(vk.maxAxis(buf, 1, false)), 0f, "maxAxis");
    assertArrayEquals(argRef, vk.download(vk.argMaxAxis(buf, 1)), 0f, "argMaxAxis");

    float[] upstream = random(outer * inner, 22);
    float[] gradRef = new float[outer * axis * inner];
    for (int o = 0; o < outer; o++) {
      for (int in = 0; in < inner; in++) {
        gradRef[(o * axis + (int) argRef[o * inner + in]) * inner + in] = upstream[o * inner + in];
      }
    }
    DeviceBuffer up = vk.upload(upstream, Shape.of(outer, inner), DType.F32, DEV);
    assertArrayEquals(gradRef, vk.download(vk.maxAxisBackward(up, buf, 1)), 0f, "maxAxisBackward");
  }

  @Test
  void sumAxis0() {
    int rows = 33, cols = 19;
    float[] a = random(rows * cols, 31);
    float[] ref = new float[cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        ref[j] += a[i * cols + j];
      }
    }
    float[] got = vk.download(vk.sumAxis0(vk.upload(a, Shape.of(rows, cols), DType.F32, DEV)));
    assertArrayEquals(ref, got, 1e-4f);
  }

  @Test
  void broadcastRank3() {
    Shape src = Shape.of(1, 4, 1);
    Shape target = Shape.of(3, 4, 5);
    float[] a = random(4, 41);
    float[] ref = new float[3 * 4 * 5];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 5; k++) {
          ref[(i * 4 + j) * 5 + k] = a[j];
        }
      }
    }
    float[] got = vk.download(vk.broadcastTo(vk.upload(a, src, DType.F32, DEV), target));
    assertArrayEquals(ref, got, 0f);
  }

  @Test
  void reluBackward() {
    int n = 500;
    float[] up = random(n, 51);
    float[] in = random(n, 52);
    float[] ref = new float[n];
    for (int i = 0; i < n; i++) {
      ref[i] = in[i] > 0 ? up[i] : 0;
    }
    DeviceBuffer dUp = vk.upload(up, Shape.of(n), DType.F32, DEV);
    DeviceBuffer dIn = vk.upload(in, Shape.of(n), DType.F32, DEV);
    assertArrayEquals(ref, vk.download(vk.reluBackward(dUp, dIn)), 0f);
  }

  @Test
  void conv2dForwardAndGrads() {
    ConvSpec spec = new ConvSpec(2, 9, 8, 3, 3, 1, 1);
    int batch = 2;
    float[] x = random(batch * spec.inputSize(), 61);
    float[] w = random(spec.outChannels() * spec.weightSize(), 62);
    int outH = spec.outHeight(), outW = spec.outWidth();

    float[] fwdRef = new float[batch * spec.outputSize()];
    for (int b = 0; b < batch; b++) {
      for (int oc = 0; oc < spec.outChannels(); oc++) {
        for (int oh = 0; oh < outH; oh++) {
          for (int ow = 0; ow < outW; ow++) {
            double acc = 0;
            for (int ic = 0; ic < spec.inChannels(); ic++) {
              for (int kh = 0; kh < spec.kernel(); kh++) {
                for (int kw = 0; kw < spec.kernel(); kw++) {
                  int ih = oh * spec.stride() - spec.pad() + kh;
                  int iw = ow * spec.stride() - spec.pad() + kw;
                  if (ih < 0 || ih >= spec.inHeight() || iw < 0 || iw >= spec.inWidth()) {
                    continue;
                  }
                  int xi = ((b * spec.inChannels() + ic) * spec.inHeight() + ih) * spec.inWidth() + iw;
                  int wi = (oc * spec.inChannels() + ic) * spec.kernel() * spec.kernel()
                      + kh * spec.kernel() + kw;
                  acc += (double) x[xi] * w[wi];
                }
              }
            }
            fwdRef[((b * spec.outChannels() + oc) * outH + oh) * outW + ow] = (float) acc;
          }
        }
      }
    }

    DeviceBuffer dx = vk.upload(x, spec.inputShape(batch), DType.F32, DEV);
    DeviceBuffer dw = vk.upload(w, spec.weightShape(), DType.F32, DEV);
    float[] fwd = vk.download(vk.conv2d(dx, dw, spec));
    assertArrayEquals(fwdRef, fwd, 1e-3f, "conv2d_fwd");

    // gradients: check they agree with a finite-difference-free direct transpose
    float[] gOut = random(batch * spec.outputSize(), 63);
    DeviceBuffer dGOut = vk.upload(gOut, spec.outputShape(batch), DType.F32, DEV);

    float[] gxRef = new float[batch * spec.inputSize()];
    float[] gwRef = new float[spec.outChannels() * spec.weightSize()];
    for (int b = 0; b < batch; b++) {
      for (int oc = 0; oc < spec.outChannels(); oc++) {
        for (int oh = 0; oh < outH; oh++) {
          for (int ow = 0; ow < outW; ow++) {
            float g = gOut[((b * spec.outChannels() + oc) * outH + oh) * outW + ow];
            for (int ic = 0; ic < spec.inChannels(); ic++) {
              for (int kh = 0; kh < spec.kernel(); kh++) {
                for (int kw = 0; kw < spec.kernel(); kw++) {
                  int ih = oh * spec.stride() - spec.pad() + kh;
                  int iw = ow * spec.stride() - spec.pad() + kw;
                  if (ih < 0 || ih >= spec.inHeight() || iw < 0 || iw >= spec.inWidth()) {
                    continue;
                  }
                  int xi = ((b * spec.inChannels() + ic) * spec.inHeight() + ih) * spec.inWidth() + iw;
                  int wi = (oc * spec.inChannels() + ic) * spec.kernel() * spec.kernel()
                      + kh * spec.kernel() + kw;
                  gxRef[xi] += g * w[wi];
                  gwRef[wi] += g * x[xi];
                }
              }
            }
          }
        }
      }
    }
    assertArrayEquals(gxRef, vk.download(vk.conv2dGradInput(dGOut, dw, spec)), 1e-3f, "conv2d_dx");
    assertArrayEquals(gwRef, vk.download(vk.conv2dGradWeight(dx, dGOut, spec)), 1e-3f, "conv2d_dw");
  }

  @Test
  void elementwiseAndScalar() {
    int n = 777;
    float[] a = random(n, 71);
    float[] b = random(n, 72);
    float[] addRef = new float[n];
    float[] mulScalarRef = new float[n];
    float[] expRef = new float[n];
    for (int i = 0; i < n; i++) {
      addRef[i] = a[i] + b[i];
      mulScalarRef[i] = a[i] * 2.5f;
      expRef[i] = (float) Math.exp(a[i]);
    }
    DeviceBuffer da = vk.upload(a, Shape.of(n), DType.F32, DEV);
    DeviceBuffer db = vk.upload(b, Shape.of(n), DType.F32, DEV);
    assertArrayEquals(addRef, vk.download(vk.binary(Op.ADD, da, db)), 1e-5f, "add");
    assertArrayEquals(mulScalarRef, vk.download(vk.scalar(Op.MUL, da, 2.5)), 1e-5f, "scalar mul");
    assertArrayEquals(expRef, vk.download(vk.unary(Op.EXP, da)), 1e-4f, "exp");
  }
}
