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
package com.nablatensor.tensor;

import com.nablatensor.tensor.spi.ComputeBackend;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.ServiceLoader;

/** Discovers {@link ComputeBackend} implementations and resolves them by selector/device. */
public final class BackendRegistry {

  private static final List<ComputeBackend> BACKENDS = load();

  private BackendRegistry() {
  }

  private static List<ComputeBackend> load() {
    List<ComputeBackend> found = new ArrayList<>();
    for (ComputeBackend backend : ServiceLoader.load(ComputeBackend.class)) {
      found.add(backend);
    }
    found.sort(Comparator.comparingInt(ComputeBackend::priority).reversed());
    return List.copyOf(found);
  }

  public static List<ComputeBackend> all() {
    return BACKENDS;
  }

  /** All backends that report themselves available on this machine. */
  public static List<ComputeBackend> available() {
    List<ComputeBackend> result = new ArrayList<>();
    for (ComputeBackend backend : BACKENDS) {
      if (backend.isAvailable()) {
        result.add(backend);
      }
    }
    return result;
  }

  /**
   * The default backend: the one named by {@code -Dnablatensor.backend=<name>} /
   * {@code NABLATENSOR_BACKEND} if that backend is present and available,
   * otherwise the highest-priority available backend (CUDA &gt; ROCm &gt;
   * Vulkan &gt; SIMD &gt; CPU).
   */
  public static ComputeBackend defaultBackend() {
    String pinned = System.getProperty("nablatensor.tensor.backend", System.getenv("NABLATENSOR_BACKEND"));
    if (pinned != null && !pinned.isBlank()) {
      String want = pinned.trim().toLowerCase(java.util.Locale.ROOT);
      for (ComputeBackend backend : BACKENDS) {
        if (backend.name().equals(want) && backend.isAvailable()) {
          return backend;
        }
      }
      // fall through to auto-selection if the pinned backend is unusable here
    }
    for (ComputeBackend backend : BACKENDS) {
      if (backend.isAvailable()) {
        return backend;
      }
    }
    throw new IllegalStateException("no nablatensor compute backend is available");
  }

  public static ComputeBackend forSelector(Backend selector) {
    return switch (selector) {
      case AUTO -> defaultBackend();
      case CPU -> forType(DeviceType.CPU);
      case CUDA -> forType(DeviceType.CUDA);
      case ROCM -> forType(DeviceType.ROCM);
      case VULKAN -> forType(DeviceType.VULKAN);
    };
  }

  public static ComputeBackend forDevice(Device device) {
    return forType(device.type());
  }

  private static ComputeBackend forType(DeviceType type) {
    for (ComputeBackend backend : BACKENDS) {
      if (backend.deviceType() == type && backend.isAvailable()) {
        return backend;
      }
    }
    throw new IllegalStateException("no available backend for device type " + type);
  }
}
