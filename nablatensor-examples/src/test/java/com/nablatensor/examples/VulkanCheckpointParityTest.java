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
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.Nabla;
import com.nablatensor.engine.vulkan.VulkanAadKernel;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.Products;
import com.nablatensor.quant.TimeGrid;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * The Vulkan kernel's checkpointed adjoint sweep and its multi-market dispatch
 * recompute or share forward values instead of keeping them; both must return
 * exactly the bits of the plain kernel, not merely agree to a tolerance. Skips
 * when no Vulkan device is present.
 */
class VulkanCheckpointParityTest {

  private static final String CKPT = "nablatensor.vulkan.ckpt";

  @BeforeAll
  static void requireVulkan() {
    assumeTrue(VulkanAadKernel.vulkanAvailable(), "no Vulkan device on this machine");
  }

  @AfterEach
  void clearProperty() {
    System.clearProperty(CKPT);
  }

  private static Nabla.TypedModel<EquityMarket> asian(int steps) {
    var grid = TimeGrid.uniform(steps);
    var product = Products.asianCall();
    return Nabla.model(EquityMarket.atmOneYear(), (rec, in) -> product.record(rec, in, grid)).fp32().greeks();
  }

  private static AadResult replay(AadTape tape, AadOptions options, double spot) {
    try (VulkanAadKernel kernel = VulkanAadKernel.compile(tape, options)) {
      kernel.setInput("spot", spot);
      return kernel.replay(20_000, 0, 42L);
    }
  }

  private static void assertSameBits(AadResult expected, AadResult actual) {
    assertEquals(expected.value(), actual.value(), 0.0);
    assertArrayEquals(expected.gradients(), actual.gradients(), 0.0);
  }

  @Test
  void checkpointedAdjointMatchesPlainKernelBitForBit() {
    var model = asian(120);   // 744 nodes: below the automatic threshold, so both variants are explicit
    AadTape tape = model.tape();
    AadOptions options = model.options();

    System.setProperty(CKPT, "1");
    AadResult plain = replay(tape, options, 101.0);
    for (String segments : new String[] {"2", "7", "16"}) {
      System.setProperty(CKPT, segments);
      assertSameBits(plain, replay(tape, options, 101.0));
    }
  }

  @Test
  void multiMarketDispatchMatchesSeparateReplays() {
    var model = asian(40);
    AadTape tape = model.tape();
    AadOptions options = model.options();
    int spotIx = tape.inputNames().indexOf("spot");
    double[] spots = {98, 100, 102};

    System.setProperty(CKPT, "4");   // checkpointing and fusion compose
    try (VulkanAadKernel kernel = VulkanAadKernel.compile(tape, options)) {
      double[][] sets = new double[spots.length][];
      AadResult[] separate = new AadResult[spots.length];
      for (int i = 0; i < spots.length; i++) {
        sets[i] = tape.recordedInputs();
        sets[i][spotIx] = spots[i];
        kernel.setInput("spot", spots[i]);
        separate[i] = kernel.replay(20_000, 0, 7L);
      }
      AadResult[] fused = kernel.replayMany(sets, 20_000, 0, 7L);
      for (int i = 0; i < spots.length; i++) {
        assertSameBits(separate[i], fused[i]);
      }
    }
  }
}
