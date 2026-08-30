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

import com.nablatensor.tensor.Device;
import com.nablatensor.tensor.NablaTensors;
import com.nablatensor.tensor.Shape;
import com.nablatensor.tensor.Tensor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * A plain 784-H-10 MLP (sigmoid hidden layer, softmax cross-entropy output)
 * trained with mini-batch SGD and <em>hand-written</em> backprop — no
 * autodiff-over-tensors, which NablaTensor deliberately does not ship. Every
 * step is a handful of {@link Tensor} ops (matmul, add, sigmoid, a stable
 * softmax, two transposed matmuls for the gradients, an SGD update) dispatched
 * to whichever {@code ComputeBackend} is available on the machine
 * (Vulkan / ROCm / CUDA); the tensor API has no CPU backend, so the example
 * prints a notice and exits cleanly when no device is present.
 *
 * <p>Point it at the classic Kaggle-format {@code mnist_train.csv}
 * ({@code label,pixel0,...,pixel783} per row) with the first CLI argument or
 * {@code -DmnistCsv=...}. With no CSV in reach it falls back to a small
 * synthetic 10-blob dataset so the pipeline still runs end to end.
 *
 * <p>Run:
 * {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.MnistMlp
 * -Dexec.args="data/mnist/mnist_train.csv"}
 *
 * <p>The numerically-stable softmax subtracts each row's maximum before
 * exponentiating and divides by an axis-aware row sum.
 */
public final class MnistMlp {

  private static final int INPUT_SIZE = 28 * 28;
  private static final int NUM_CLASSES = 10;
  private static final double TRAIN_FRACTION = 0.8;

  private MnistMlp() {
  }

  public static void main(String[] args) throws IOException {
    String csvPath = args.length > 0 ? args[0] : System.getProperty("mnistCsv", "data/mnist/mnist_train.csv");
    int epochs = Integer.getInteger("epochs", 30);
    int batchSize = Integer.getInteger("batch", 100);
    int hidden = Integer.getInteger("hidden", 256);
    double lr = Double.parseDouble(System.getProperty("lr", "1.5"));

    if (NablaTensors.devices().isEmpty()) {
      System.out.println("MnistMlp needs a tensor compute backend (Vulkan / ROCm / CUDA); "
          + "none is available on this machine. Skipping.");
      return;
    }
    Device device = NablaTensors.defaultDevice();
    System.out.println("nablatensor devices: " + NablaTensors.devices() + " -> training on " + device);

    Random random = new Random(0);
    long loadStart = System.nanoTime();
    Dataset data = loadOrSynthesize(csvPath, random);
    long loadMs = (System.nanoTime() - loadStart) / 1_000_000;
    System.out.printf(Locale.ROOT, "%s: %d samples in %d ms%n", data.source, data.count, loadMs);

    // Fixed 80/20 split on row order: the first TRAIN_FRACTION rows train, the rest test.
    int trainCount = (int) Math.round(data.count * TRAIN_FRACTION);
    int testCount = data.count - trainCount;
    System.out.printf(Locale.ROOT, "split: %d train / %d test (80/20 fixed)%n", trainCount, testCount);
    int[] trainIndices = new int[trainCount];
    for (int i = 0; i < trainCount; i++) {
      trainIndices[i] = i;
    }
    int[] testIndices = new int[testCount];
    for (int i = 0; i < testCount; i++) {
      testIndices[i] = trainCount + i;
    }

    // Scale each weight by 1/sqrt(fan-in) so the sigmoids start in their linear
    // middle rather than saturated flat where the gradient vanishes — this is
    // what lets the plain SGD loop reach ~98% instead of stalling near 94%.
    Tensor w1 = scaleAndClose(NablaTensors.randn(random, INPUT_SIZE, hidden), 1.0 / Math.sqrt(INPUT_SIZE));
    Tensor b1 = NablaTensors.zeros(hidden);
    Tensor w2 = scaleAndClose(NablaTensors.randn(random, hidden, NUM_CLASSES), 1.0 / Math.sqrt(hidden));
    Tensor b2 = NablaTensors.zeros(NUM_CLASSES);

    long trainStart = System.nanoTime();
    for (int epoch = 0; epoch < epochs; epoch++) {
      shuffle(trainIndices, random);
      double epochLoss = 0.0;
      int batches = 0;
      for (int start = 0; start < trainCount; start += batchSize) {
        int end = Math.min(start + batchSize, trainCount);
        int bs = end - start;
        Tensor x = uploadBatch(data.images, trainIndices, start, bs, device);
        Tensor y = uploadOneHot(data.labels, trainIndices, start, bs, device);

        Step step = trainStep(w1, b1, w2, b2, x, y, bs, lr);
        x.close();
        y.close();
        w1.close();
        b1.close();
        w2.close();
        b2.close();
        w1 = step.w1;
        b1 = step.b1;
        w2 = step.w2;
        b2 = step.b2;
        epochLoss += step.loss;
        batches++;
      }
      System.out.printf(Locale.ROOT, "epoch %d/%d  avg loss %.4f%n", epoch + 1, epochs, epochLoss / batches);
    }
    long trainMs = (System.nanoTime() - trainStart) / 1_000_000;
    double imgPerSec = trainMs == 0 ? 0 : epochs * (double) trainCount / (trainMs / 1000.0);
    System.out.printf(Locale.ROOT, "training: %d epochs x %d samples in %d ms (%.0f img/s)%n",
        epochs, trainCount, trainMs, imgPerSec);

    double trainAccuracy = evaluate(w1, b1, w2, b2, data, trainIndices, trainCount, batchSize, device);
    double testAccuracy = evaluate(w1, b1, w2, b2, data, testIndices, testCount, batchSize, device);
    System.out.printf(Locale.ROOT, "final train accuracy: %.2f%%, test accuracy: %.2f%%%n",
        trainAccuracy * 100, testAccuracy * 100);

    w1.close();
    b1.close();
    w2.close();
    b2.close();
  }

  // ---- one training step (forward + manual backprop + SGD update) ---------

  private record Step(Tensor w1, Tensor b1, Tensor w2, Tensor b2, float loss) {
  }

  private static Step trainStep(Tensor w1, Tensor b1, Tensor w2, Tensor b2,
      Tensor x, Tensor y, int batchSize, double lr) {
    // forward: a1 = sigmoid(x @ w1 + b1)
    Tensor z1raw = x.matmul(w1);
    Tensor z1 = z1raw.add(b1);
    z1raw.close();
    Tensor a1 = z1.sigmoid();
    z1.close();

    // logits: z2 = a1 @ w2 + b2
    Tensor z2raw = a1.matmul(w2);
    Tensor z2 = z2raw.add(b2);
    z2raw.close();

    // stable softmax: probs = softmax(z2 - rowmax(z2))
    Tensor maxZ2 = z2.max(1, true);
    Tensor z2shifted = z2.sub(maxZ2);
    maxZ2.close();
    z2.close();
    Tensor expZ = z2shifted.exp();
    z2shifted.close();
    Tensor rowSum = expZ.sum(1, true);
    Tensor probs = expZ.div(rowSum);
    expZ.close();
    rowSum.close();

    // cross-entropy loss (reporting only)
    Tensor logProbs = probs.log();
    Tensor yLogProbs = y.mul(logProbs);
    logProbs.close();
    Tensor lossSum = yLogProbs.sum();
    yLogProbs.close();
    float loss = -lossSum.item() / batchSize;
    lossSum.close();

    // backward: dZ2 = (probs - y) / batchSize
    Tensor dZ2raw = probs.sub(y);
    probs.close();
    Tensor dZ2 = dZ2raw.mul(1.0 / batchSize);
    dZ2raw.close();

    Tensor a1T = a1.transpose();
    Tensor dW2 = a1T.matmul(dZ2);
    a1T.close();
    Tensor db2 = dZ2.sumAxis0();

    Tensor w2T = w2.transpose();
    Tensor dA1 = dZ2.matmul(w2T);
    w2T.close();
    dZ2.close();

    // sigmoid'(z1) = a1 * (1 - a1)
    Tensor negA1 = a1.neg();
    Tensor oneMinusA1 = negA1.add(1.0);
    negA1.close();
    Tensor sigDeriv = a1.mul(oneMinusA1);
    oneMinusA1.close();
    a1.close();
    Tensor dZ1 = dA1.mul(sigDeriv);
    dA1.close();
    sigDeriv.close();

    Tensor xT = x.transpose();
    Tensor dW1 = xT.matmul(dZ1);
    xT.close();
    Tensor db1 = dZ1.sumAxis0();
    dZ1.close();

    return new Step(
        update(w1, dW1, lr), update(b1, db1, lr),
        update(w2, dW2, lr), update(b2, db2, lr), loss);
  }

  /** Returns {@code param - lr * grad}, closing {@code grad} and its scaled intermediate. */
  private static Tensor update(Tensor param, Tensor grad, double lr) {
    Tensor scaled = grad.mul(lr);
    grad.close();
    Tensor updated = param.sub(scaled);
    scaled.close();
    return updated;
  }

  // ---- evaluation (forward-only, argmax on the host) ----------------------

  private static double evaluate(Tensor w1, Tensor b1, Tensor w2, Tensor b2,
      Dataset data, int[] indices, int count, int batchSize, Device device) {
    if (count == 0) {
      return 0.0;
    }
    long correct = 0;
    for (int start = 0; start < count; start += batchSize) {
      int end = Math.min(start + batchSize, count);
      int bs = end - start;
      Tensor x = uploadBatch(data.images, indices, start, bs, device);

      Tensor z1raw = x.matmul(w1);
      Tensor z1 = z1raw.add(b1);
      z1raw.close();
      Tensor a1 = z1.sigmoid();
      z1.close();
      Tensor z2raw = a1.matmul(w2);
      a1.close();
      Tensor z2 = z2raw.add(b2);
      z2raw.close();
      x.close();

      float[] logits = z2.toFloatArray();
      z2.close();
      for (int i = 0; i < bs; i++) {
        int base = i * NUM_CLASSES;
        int argmax = 0;
        float best = logits[base];
        for (int c = 1; c < NUM_CLASSES; c++) {
          if (logits[base + c] > best) {
            best = logits[base + c];
            argmax = c;
          }
        }
        if (argmax == data.labels[indices[start + i]]) {
          correct++;
        }
      }
    }
    return correct / (double) count;
  }

  // ---- batch upload helpers ---------------------------------------------------

  private static Tensor uploadBatch(float[] images, int[] order, int start, int bs, Device device) {
    float[] batch = new float[bs * INPUT_SIZE];
    for (int i = 0; i < bs; i++) {
      System.arraycopy(images, order[start + i] * INPUT_SIZE, batch, i * INPUT_SIZE, INPUT_SIZE);
    }
    return NablaTensors.arrayOn(batch, Shape.of(bs, INPUT_SIZE), device);
  }

  private static Tensor uploadOneHot(int[] labels, int[] order, int start, int bs, Device device) {
    float[] batch = new float[bs * NUM_CLASSES];
    for (int i = 0; i < bs; i++) {
      batch[i * NUM_CLASSES + labels[order[start + i]]] = 1f;
    }
    return NablaTensors.arrayOn(batch, Shape.of(bs, NUM_CLASSES), device);
  }

  private static Tensor scaleAndClose(Tensor t, double scale) {
    Tensor scaled = t.mul(scale);
    t.close();
    return scaled;
  }

  private static void shuffle(int[] order, Random random) {
    for (int i = order.length - 1; i > 0; i--) {
      int j = random.nextInt(i + 1);
      int tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }
  }

  // ---- data: real CSV if present, else a synthetic 10-blob set --------------

  private record Dataset(float[] images, int[] labels, int count, String source) {
  }

  private static Dataset loadOrSynthesize(String csvPath, Random random) throws IOException {
    Path path = resolvePath(csvPath);
    if (path != null) {
      return loadCsv(path);
    }
    System.out.printf(Locale.ROOT, "no MNIST CSV at '%s' — using a synthetic 10-blob dataset%n", csvPath);
    return synthesize(2_000, random);
  }

  private static Path resolvePath(String csvPath) {
    for (Path candidate : new Path[] {Path.of(csvPath), Path.of("..").resolve(csvPath)}) {
      if (Files.exists(candidate)) {
        return candidate;
      }
    }
    return null;
  }

  /**
   * {@code count} samples drawn from {@link #NUM_CLASSES} isotropic Gaussian
   * blobs in the 784-dim pixel cube — linearly separable enough that the MLP
   * reaches high accuracy in a few epochs, so the training loop exercises every
   * op path without a data download.
   */
  private static Dataset synthesize(int count, Random random) {
    float[][] centres = new float[NUM_CLASSES][INPUT_SIZE];
    for (int c = 0; c < NUM_CLASSES; c++) {
      for (int p = 0; p < INPUT_SIZE; p++) {
        centres[c][p] = random.nextFloat();
      }
    }
    float[] images = new float[count * INPUT_SIZE];
    int[] labels = new int[count];
    for (int i = 0; i < count; i++) {
      int c = random.nextInt(NUM_CLASSES);
      labels[i] = c;
      int base = i * INPUT_SIZE;
      for (int p = 0; p < INPUT_SIZE; p++) {
        float v = centres[c][p] + 0.15f * (float) random.nextGaussian();
        images[base + p] = Math.max(0f, Math.min(1f, v));
      }
    }
    return new Dataset(images, labels, count, "synthetic 10-blob");
  }

  /** Parses "label,pixel0,...,pixel783" lines without regex splitting, for speed. */
  private static Dataset loadCsv(Path path) throws IOException {
    List<int[]> labelChunks = new ArrayList<>();
    List<float[]> imageChunks = new ArrayList<>();
    int count = 0;
    final int chunk = 65_536;
    try (BufferedReader reader = new BufferedReader(new FileReader(path.toFile()), 1 << 20)) {
      String line;
      int[] labels = new int[chunk];
      float[] images = new float[chunk * INPUT_SIZE];
      while ((line = reader.readLine()) != null) {
        if (line.isEmpty()) {
          continue;
        }
        if (count > 0 && count % chunk == 0) {
          labelChunks.add(labels);
          imageChunks.add(images);
          labels = new int[chunk];
          images = new float[chunk * INPUT_SIZE];
        }
        int slot = count % chunk;
        int base = slot * INPUT_SIZE;
        int len = line.length();
        int col = 0;
        int value = 0;
        for (int i = 0; i <= len; i++) {
          char ch = i < len ? line.charAt(i) : ',';
          if (ch == ',') {
            if (col == 0) {
              labels[slot] = value;
            } else if (col <= INPUT_SIZE) {
              images[base + col - 1] = value / 255f;
            }
            col++;
            value = 0;
          } else {
            value = value * 10 + (ch - '0');
          }
        }
        count++;
      }
      labelChunks.add(labels);
      imageChunks.add(images);
    }

    int[] allLabels = new int[count];
    float[] allImages = new float[count * INPUT_SIZE];
    int written = 0;
    for (int c = 0; c < labelChunks.size(); c++) {
      int chunkCount = Math.min(chunk, count - written);
      System.arraycopy(labelChunks.get(c), 0, allLabels, written, chunkCount);
      System.arraycopy(imageChunks.get(c), 0, allImages, written * INPUT_SIZE, chunkCount * INPUT_SIZE);
      written += chunkCount;
    }
    return new Dataset(allImages, allLabels, count, path.toString());
  }
}
