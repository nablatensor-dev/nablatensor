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

import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.Products;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Generates a model-validation evidence pack for the Asian call and, if a path
 * is given, writes it there.
 *
 * <p>Run: {@code mvn -o -q -pl nablatensor-validate exec:java
 * -Dexec.mainClass=com.nablatensor.validate.EvidenceMain}
 */
public final class EvidenceMain {

  private EvidenceMain() {
  }

  public static void main(String[] args) throws IOException {
    Report report = ModelValidation.of(Products.asianCall())
        .market(EquityMarket.atmOneYear())
        .steps(Integer.getInteger("steps", 252))
        .scenarios(Long.getLong("scenarios", 1_000_000L))
        .seed(Long.getLong("seed", 42L))
        .fp64()
        .tolerance(1e-6)
        .run();

    System.out.println(report);
    if (args.length > 0) {
      report.writeText(Path.of(args[0]));
      System.out.println("written to " + args[0]);
    }
    if (!report.passed()) {
      System.exit(1);
    }
  }
}
