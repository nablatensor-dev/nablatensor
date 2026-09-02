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
package com.nablatensor.tensor.spi;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * One GPU kernel: its CUDA-C source plus the launch geometry it was written for.
 *
 * <p>The entry-point name and parameter types are never written by hand — {@link
 * #of(String, int, int)} reads them out of the source, so the name a backend
 * passes to {@code cuModuleGetFunction} cannot drift from the name the source
 * actually defines.
 */
public record GpuKernel(String name, List<String> paramTypes, String source,
                        int blockDimX, int blockDimY) {

  /** Block size the flat one-dimensional kernels are written for. */
  public static final int DEFAULT_BLOCK_DIM = 256;

  private static final Pattern ENTRY_POINT = Pattern.compile(
      "extern\\s+\"C\"\\s+__global__\\s+void\\s+(\\w+)\\s*\\(([^)]*)\\)", Pattern.DOTALL);

  private static final Pattern PARAMETER = Pattern.compile("(.*?)([A-Za-z_]\\w*)", Pattern.DOTALL);

  public GpuKernel {
    Objects.requireNonNull(name, "name");
    Objects.requireNonNull(source, "source");
    paramTypes = List.copyOf(paramTypes);
    if (blockDimX < 1 || blockDimY < 1) {
      throw new IllegalArgumentException(name + ": block dims must be positive");
    }
  }

  /** Parses a single-entry-point translation unit launched with the default block. */
  public static GpuKernel of(String source) {
    return of(source, DEFAULT_BLOCK_DIM, 1);
  }

  /**
   * Parses a translation unit that must declare exactly one {@code extern "C"
   * __global__} entry point; {@code __device__} helpers alongside it are fine.
   */
  public static GpuKernel of(String source, int blockDimX, int blockDimY) {
    Matcher matcher = ENTRY_POINT.matcher(source);
    if (!matcher.find()) {
      throw new IllegalArgumentException(
          "no extern \"C\" __global__ entry point in: " + excerpt(source));
    }
    String name = matcher.group(1);
    List<String> paramTypes = parseParamTypes(name, matcher.group(2));
    if (matcher.find()) {
      throw new IllegalArgumentException(
          name + ": one GpuKernel must declare one entry point, also found " + matcher.group(1));
    }
    return new GpuKernel(name, paramTypes, source, blockDimX, blockDimY);
  }

  /** Number of arguments a launch of this kernel must supply. */
  public int arity() {
    return paramTypes.size();
  }

  private static List<String> parseParamTypes(String kernel, String parameterList) {
    List<String> types = new ArrayList<>();
    for (String parameter : parameterList.split(",")) {
      String declaration = parameter.trim().replaceAll("\\s+", " ");
      Matcher matcher = PARAMETER.matcher(declaration);
      if (!matcher.matches() || matcher.group(1).isBlank()) {
        throw new IllegalArgumentException(
            kernel + ": cannot read a type and a name from parameter '" + declaration + "'");
      }
      types.add(matcher.group(1).trim());
    }
    return types;
  }

  private static String excerpt(String source) {
    String head = source.strip();
    return head.length() <= 80 ? head : head.substring(0, 80) + "...";
  }
}
