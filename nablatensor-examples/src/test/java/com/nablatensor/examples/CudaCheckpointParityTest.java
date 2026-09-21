package com.nablatensor.examples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.nablatensor.engine.ADouble;
import com.nablatensor.engine.AadCheckpointPlan;
import com.nablatensor.engine.AadExecutable;
import com.nablatensor.engine.AadOptions;
import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.AadResult;
import com.nablatensor.engine.AadTape;
import com.nablatensor.engine.CudaAadCodegen;
import com.nablatensor.engine.CudaAadEngine;
import com.nablatensor.engine.DeviceAadExecutable;
import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.Products;
import com.nablatensor.quant.TimeGrid;
import org.junit.jupiter.api.Test;

class CudaCheckpointParityTest {

  private static final String CHECKPOINT = "nablatensor.checkpoint";
  private static final String MIN_NODES = "nablatensor.checkpoint.minNodes";
  private static final String SEGMENT_LENGTH = "nablatensor.checkpoint.segLen";

  @Test
  void opaqueCodegenIsCudaOnly() {
    AadTape tape = asian(120);
    AadOptions options = AadOptions.defaults();
    AadCheckpointPlan plan = AadCheckpointPlan.of(tape, options, 2);
    assertNotNull(plan);
    String portable = CudaAadCodegen.generateCheckpointed(tape, options, plan);
    assertFalse(portable.contains("asm volatile"));
    assertFalse(portable.contains("__launch_bounds__"));
    for (AadOptions.PrecisionEnum precision : AadOptions.PrecisionEnum.values()) {
      String cuda = CudaAadCodegen.generateCheckpointed(tape, options.withPrecision(precision), plan, true);
      assertTrue(cuda.contains("asm volatile"));
      assertTrue(cuda.contains("volatile real* __restrict__ scratch"));
      assertTrue(cuda.contains("__launch_bounds__(BLOCK, 4)"));
    }
    String boundsOnly = CudaAadCodegen.generateCheckpointed(tape, options, plan, false, false, 2);
    assertTrue(boundsOnly.contains("__launch_bounds__(BLOCK, 2)"));
    assertFalse(boundsOnly.contains("asm volatile"));
    assertFalse(boundsOnly.contains("volatile real*"));
    String barriersOnly = CudaAadCodegen.generateCheckpointed(tape, options, plan, true, true, 0);
    assertFalse(barriersOnly.contains("__launch_bounds__"));
    assertTrue(barriersOnly.contains("asm volatile"));
    assertThrows(IllegalArgumentException.class,
        () -> CudaAadCodegen.generateCheckpointed(tape, options, plan, true, true, -1));
    assertThrows(IllegalArgumentException.class,
        () -> CudaAadCodegen.generateCheckpointed(tape, options, plan, true, true, 9));
  }

  @Test
  void fp32AsianMatchesPlain() {
    assertParity(asian(120), AadOptions.defaults(), "spot", 101.0, 48);
  }

  @Test
  void tunedSegmentsMatchPlain() {
    assertParity(asian(120), AadOptions.defaults(), "spot", 101.0, 96);
  }

  @Test
  void fp64AsianMatchesPlain() {
    assertParity(asian(32), AadOptions.defaults().withPrecision(AadOptions.PrecisionEnum.FLOAT64),
        "spot", 98.0, 17);
  }

  @Test
  void randomBranchingTapeMatchesPlain() {
    AadTape tape = AadRecorder.record(recorder -> {
      ADouble level = recorder.input("level", 1.2);
      ADouble scale = recorder.input("scale", 0.1);
      for (int step = 0; step < 24; step++) {
        level = level.add(recorder.randn().mul(scale)).add(recorder.randn().mul(0.01)).max(0.01);
      }
      recorder.output(level.log().exp().sqrt().div(scale.add(1.0)));
    });
    assertParity(tape, AadOptions.defaults(), "level", 1.5, 19);
  }

  private static AadTape asian(int steps) {
    var product = Products.asianCall();
    var grid = TimeGrid.uniform(steps);
    return Nabla.model(EquityMarket.atmOneYear(),
        (recorder, inputs) -> product.record(recorder, inputs, grid)).tape();
  }

  private static void assertParity(AadTape tape, AadOptions options, String input,
                                   double shifted, int segmentLength) {
    CudaAadEngine engine = new CudaAadEngine();
    assumeTrue(engine.isAvailable(), "no CUDA device on this machine");
    String oldMode = System.getProperty(CHECKPOINT);
    String oldMin = System.getProperty(MIN_NODES);
    String oldLength = System.getProperty(SEGMENT_LENGTH);
    try {
      System.setProperty(CHECKPOINT, "off");
      try (AadExecutable plain = engine.compile(tape, options)) {
        System.setProperty(CHECKPOINT, "on");
        System.setProperty(MIN_NODES, "2");
        System.setProperty(SEGMENT_LENGTH, Integer.toString(segmentLength));
        assertNotNull(AadCheckpointPlan.of(tape, options, DeviceAadExecutable.checkpointMinNodes()));
        try (AadExecutable checkpointed = engine.compile(tape, options)) {
          for (long seed : new long[] {42L, 0x12345678fedcba98L}) {
            for (long paths : new long[] {513L, 20_000L}) {
              assertSame(plain.replay(paths, 0x100000003L, seed),
                  checkpointed.replay(paths, 0x100000003L, seed), options.precision());
            }
            plain.setInput(input, shifted);
            checkpointed.setInput(input, shifted);
          }
        }
      }
    } finally {
      restore(CHECKPOINT, oldMode);
      restore(MIN_NODES, oldMin);
      restore(SEGMENT_LENGTH, oldLength);
    }
  }

  private static void assertSame(AadResult expected, AadResult actual, AadOptions.PrecisionEnum precision) {
    double tolerance = precision == AadOptions.PrecisionEnum.FLOAT32 ? 2e-6 : 2e-12;
    assertClose(expected.value(), actual.value(), tolerance);
    double[] expectedGradients = expected.gradients();
    double[] actualGradients = actual.gradients();
    assertEquals(expectedGradients.length, actualGradients.length);
    for (int index = 0; index < expectedGradients.length; index++) {
      assertClose(expectedGradients[index], actualGradients[index], tolerance);
    }
  }

  private static void assertClose(double expected, double actual, double tolerance) {
    assertTrue(Double.isFinite(expected));
    assertTrue(Double.isFinite(actual));
    assertEquals(expected, actual, tolerance * Math.max(1.0, Math.abs(expected)));
  }

  private static void restore(String name, String value) {
    if (value == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, value);
    }
  }
}