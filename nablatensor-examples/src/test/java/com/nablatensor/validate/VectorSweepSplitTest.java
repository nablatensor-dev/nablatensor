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
package com.nablatensor.validate;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.lang.classfile.ClassFile;
import java.lang.classfile.attribute.CodeAttribute;
import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

/**
 * The double-precision sweeps must stay in methods of their own.
 *
 * <p>They used to be two loops inside {@code runRange}, which made it 1,023
 * bytes of bytecode; inlining fourteen opcode helpers and the generator into a
 * body that size exhausts what C2 will inline in one go, and the rest is
 * deferred to incremental inlining. Vector API calls resolved that late have
 * lost the constant folding they depend on — the operator argument of
 * {@code lanewise} is no longer a constant when the branch routing
 * transcendentals to the vector math library is parsed, so {@code EXP} takes
 * the general path and reaches code that does not expect a math-library
 * opcode. On JDK 25 that aborts the compiler thread: about one fresh JVM in
 * four died partway through the adjoint sweep. Short of the crash it costs
 * throughput and makes it depend on which compilation happened to win, which
 * is how the fault first showed up — a benchmark row that had to be quoted as
 * a range.
 *
 * <p>What is pinned here is that split, not the throughput it bought. A timing
 * assertion would measure the build machine; this reads the class file, so it
 * holds on the no-incubator CI path too, where the Vector API is absent.
 */
class VectorSweepSplitTest {

  /**
   * Comfortably above the driver loop as it stands (280 bytes) and far below
   * the 1,023 that broke. The number is a tripwire, not a budget: any change
   * big enough to cross it has folded a sweep back into the driver.
   */
  private static final int DRIVER_LIMIT = 512;

  @Test
  void theDoublePrecisionDriverDoesNotCarryTheSweeps() throws IOException {
    Map<String, Integer> sizes =
        methodSizes("com/nablatensor/engine/simd/VectorReplayF64.class");

    assertTrue(sizes.containsKey("forward") && sizes.containsKey("reverse"),
        "VectorReplayF64 must keep its forward and reverse sweeps in methods of their own,"
            + " found: " + sizes.keySet());

    int driver = sizes.get("runRange");
    assertTrue(driver <= DRIVER_LIMIT,
        "VectorReplayF64.runRange is " + driver + " bytes of bytecode, over the "
            + DRIVER_LIMIT + "-byte tripwire — a sweep has been folded back into the driver,"
            + " which late-inlines the Vector API calls it holds and crashes C2 on lanewise(EXP)");
  }

  private static Map<String, Integer> methodSizes(String resource) throws IOException {
    try (InputStream in =
        VectorSweepSplitTest.class.getClassLoader().getResourceAsStream(resource)) {
      assertTrue(in != null, resource + " is not on the test classpath");
      Map<String, Integer> sizes = new LinkedHashMap<>();
      for (var method : ClassFile.of().parse(in.readAllBytes()).methods()) {
        method.findAttribute(java.lang.classfile.Attributes.code()).ifPresent(code ->
            sizes.put(method.methodName().stringValue(), ((CodeAttribute) code).codeLength()));
      }
      return sizes;
    }
  }
}
