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
package com.nablatensor.backend.wgpu;

/**
 * Public, tensor-free entry point to the headless wgpu-native compute
 * runtime. Mirrors {@code VulkanCompute}; the one shape difference is that a
 * WebGPU buffer is a single opaque handle (no separate device-memory object
 * to pair it with), and there is no push-constant block — the caller's last
 * buffer is a small read-only storage buffer instead, refreshed from
 * {@code pushInts} before each dispatch.
 */
public final class WgpuCompute {

  private WgpuCompute() {
  }

  /** Whether a WebGPU adapter and device could be brought up on this machine. */
  public static boolean isAvailable() {
    return WgpuRuntime.probe();
  }

  /** Name of the adapter the runtime selected, plus its backend (Vulkan, Metal, ...). */
  public static String deviceName() {
    return WgpuRuntime.deviceName();
  }

  /**
   * Builds a compute pipeline named {@code name} from {@code wgsl}, with one
   * read-only or read-write storage buffer binding per entry of
   * {@code readOnly}. A no-op if {@code name} is already registered, so the
   * caller keys distinct tapes by distinct names.
   */
  public static void registerPipeline(String name, String wgsl, boolean[] readOnly) {
    WgpuRuntime.registerPipeline(name, wgsl, readOnly);
  }

  /** Handle for a device buffer of {@code bytes}, usable as both a dispatch binding and a copy source/destination. */
  public static long alloc(long bytes) {
    return WgpuRuntime.alloc(bytes);
  }

  public static void free(long buffer) {
    WgpuRuntime.free(buffer);
  }

  public static void writeFloats(long buffer, float[] data) {
    WgpuRuntime.writeFloats(buffer, data);
  }

  public static float[] readFloats(long buffer, int count) {
    return WgpuRuntime.readFloats(buffer, count);
  }

  /**
   * Submits one dispatch of {@code groupsX} workgroups over {@code buffers}
   * (bound {@code 0..n-1}), with {@code pushInts} written into the last
   * buffer first, and blocks until it retires.
   */
  public static void dispatch(String kernel, int groupsX, long[] buffers, int[] pushInts) {
    WgpuRuntime.dispatch(kernel, groupsX, 1, 1, buffers, pushInts);
  }

  public static void sync() {
    WgpuRuntime.sync();
  }
}
