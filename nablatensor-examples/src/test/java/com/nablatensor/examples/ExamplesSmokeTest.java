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

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.nablatensor.tensor.NablaTensors;
import org.junit.jupiter.api.Test;

/**
 * Every example is also a test: each {@code main} must run to completion on a
 * small scenario count. Keeps the docs pages from bit-rotting.
 */
class ExamplesSmokeTest {

  @Test
  void vanillaEuropean() {
    System.setProperty("scenarios", "50000");
    VanillaEuropeanGreeks.main(new String[0]);
  }

  @Test
  void asianAcrossBackends() {
    System.setProperty("scenarios", "50000");
    System.setProperty("steps", "32");
    AsianGreeksBackends.main(new String[0]);
  }

  @Test
  void swapThePayoff() {
    System.setProperty("scenarios", "50000");
    System.setProperty("steps", "32");
    SwapThePayoff.main(new String[0]);
  }

  @Test
  void sabrCalibration() {
    HestonSabrCalibration.main(new String[0]);
  }

  @Test
  void mnistMlp() throws Exception {
    assumeTrue(!NablaTensors.devices().isEmpty(),
        "MnistMlp needs a tensor compute backend (Vulkan / ROCm / CUDA)");
    System.setProperty("epochs", "5");
    System.setProperty("batch", "64");
    System.setProperty("hidden", "32");
    System.setProperty("mnistCsv", "does-not-exist.csv");  // force the synthetic fallback
    MnistMlp.main(new String[0]);
  }
}
