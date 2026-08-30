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
package com.nablatensor.ops;

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.SDouble;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A registry for user-named operations, in <em>macro</em> form: a custom op is a
 * function that expands into primitive {@link SDouble} nodes when it is
 * recorded. Its adjoint is whatever the recorded sub-graph produces, so it works
 * on every backend with no engine change (Seam 3, composable subset).
 *
 * <pre>{@code
 * CustomOp.registerUnary("softplus", (rec, x) -> Smooth.ramp(rec, x, 0.05));
 * SDouble y = CustomOp.unary("softplus").apply(rec, x);
 * }</pre>
 *
 * <p>A fused op with a hand-written {@code {forward(x), adjoint(x, gbar)}} and
 * per-backend code generation — for kernels that cannot be expressed in the
 * primitive set at all — is a later engine feature. For everything expressible
 * as a composition (which is most of what a quant registers) this form is
 * equivalent and needs nothing from the engine.
 */
public final class CustomOp {

  /** A named op of one tape argument. */
  @FunctionalInterface
  public interface Unary {
    SDouble apply(AadRecorder rec, SDouble x);
  }

  /** A named op of two tape arguments. */
  @FunctionalInterface
  public interface Binary {
    SDouble apply(AadRecorder rec, SDouble a, SDouble b);
  }

  private static final Map<String, Unary> UNARY = new ConcurrentHashMap<>();
  private static final Map<String, Binary> BINARY = new ConcurrentHashMap<>();

  static {
    registerUnary("relu", (rec, x) -> x.max(0.0));
    registerUnary("softplus", (rec, x) -> Smooth.ramp(rec, x, 0.05));
    registerUnary("sigmoid", (rec, x) -> Smooth.step(rec, x, 1.0));
    registerUnary("normCdf", SpecialFn::normCdf);
  }

  private CustomOp() {
  }

  public static void registerUnary(String name, Unary op) {
    UNARY.put(requireName(name), Objects.requireNonNull(op, "op"));
  }

  public static void registerBinary(String name, Binary op) {
    BINARY.put(requireName(name), Objects.requireNonNull(op, "op"));
  }

  public static Unary unary(String name) {
    Unary op = UNARY.get(name);
    if (op == null) {
      throw new IllegalArgumentException("no unary custom op named '" + name + "'; registered: " + UNARY.keySet());
    }
    return op;
  }

  public static Binary binary(String name) {
    Binary op = BINARY.get(name);
    if (op == null) {
      throw new IllegalArgumentException("no binary custom op named '" + name + "'; registered: " + BINARY.keySet());
    }
    return op;
  }

  public static boolean hasUnary(String name) {
    return UNARY.containsKey(name);
  }

  private static String requireName(String name) {
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("custom op name must be non-blank");
    }
    return name;
  }
}
