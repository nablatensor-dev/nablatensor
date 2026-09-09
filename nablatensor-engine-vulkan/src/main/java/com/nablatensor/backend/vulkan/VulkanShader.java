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
package com.nablatensor.backend.vulkan;

import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * One GLSL compute shader: the pipeline name it registers under and its source.
 *
 * <p>The {@code std430} storage-buffer count a pipeline is built with — and that
 * a {@code dispatch} must then supply buffers for — is never written by hand:
 * {@link #of} reads it out of the source as the highest {@code layout(..., binding
 * = N) buffer} plus one, so it cannot drift from the {@code buffer} blocks the
 * shader actually declares. A GLSL compute shader has no user-visible entry-point
 * name (it is always {@code main}), so {@code name} is supplied by the caller as
 * the {@link VulkanRuntime#registerPipeline registration} key.
 */
record VulkanShader(String name, String source, int bindings) {

  /** {@code binding = N} ... {@code buffer} on one {@code layout(...)} line, no {@code ;}/{@code {} between. */
  private static final Pattern STORAGE_BINDING =
      Pattern.compile("binding\\s*=\\s*(\\d+)[^;{]*?\\bbuffer\\b");

  VulkanShader {
    Objects.requireNonNull(name, "name");
    Objects.requireNonNull(source, "source");
    if (bindings < 1) {
      throw new IllegalArgumentException(name + ": a compute shader needs at least one storage buffer");
    }
  }

  /** Parses {@code source} for its {@code std430 buffer} bindings and keys it by {@code name}. */
  static VulkanShader of(String name, String source) {
    Matcher matcher = STORAGE_BINDING.matcher(source);
    int maxBinding = -1;
    while (matcher.find()) {
      maxBinding = Math.max(maxBinding, Integer.parseInt(matcher.group(1)));
    }
    if (maxBinding < 0) {
      throw new IllegalArgumentException(
          name + ": no 'layout(..., binding = N) buffer' declaration in source");
    }
    return new VulkanShader(name, source, maxBinding + 1);
  }
}
