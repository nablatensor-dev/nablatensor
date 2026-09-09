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

/**
 * Public, tensor-free entry point to the headless Vulkan compute runtime.
 *
 * <p>{@link VulkanRuntime} is package-private and its only public face is
 * {@link VulkanBackend}, which is bound to the {@code nablatensor-core} tensor SPI.
 * A caller that wants to register and dispatch its own compute shaders —
 * {@code nablatensor-vulkan}'s replay kernel being the motivating case — has no
 * need of tensors, so this class re-exports just the raw primitives:
 * shader registration, buffer allocation, host copies, and a one-dimensional
 * dispatch that blocks on {@code vkQueueWaitIdle}.
 */
public final class VulkanCompute {

  private VulkanCompute() {
  }

  /** Whether a Vulkan compute device could be brought up on this machine. */
  public static boolean isAvailable() {
    return VulkanRuntime.probe();
  }

  /** Name of the physical device the runtime selected. */
  public static String deviceName() {
    return VulkanRuntime.deviceName();
  }

  /**
   * Compiles {@code glsl} (a {@code compute} shader) to SPIR-V and builds a
   * pipeline for it, bound to {@code bindings} {@code std430} storage buffers at
   * bindings {@code 0 .. bindings-1} plus a 128-byte compute push-constant
   * block. A no-op if {@code name} is already registered, so the caller keys
   * distinct tapes by distinct names.
   */
  public static void registerPipeline(String name, String glsl, int bindings) {
    VulkanRuntime.registerPipeline(name, glsl, bindings);
  }

  /** {@code [buffer, memory]} handles for a device buffer of {@code bytes}. */
  public static long[] alloc(long bytes) {
    return VulkanRuntime.alloc(bytes);
  }

  public static void free(long buffer, long memory) {
    VulkanRuntime.free(buffer, memory);
  }

  public static void writeFloats(long memory, float[] data) {
    VulkanRuntime.writeFloats(memory, data);
  }

  public static float[] readFloats(long memory, int count) {
    return VulkanRuntime.readFloats(memory, count);
  }

  /**
   * Records and submits one dispatch of {@code groupsX} workgroups over
   * {@code buffers} (bound 0..n-1), with {@code pushInts} written into the
   * push-constant block, and blocks until it retires.
   */
  public static void dispatch(String kernel, int groupsX, long[] buffers, int[] pushInts) {
    VulkanRuntime.dispatch(kernel, groupsX, 1, 1, buffers, pushInts, -1, 0f);
  }

  public static void sync() {
    VulkanRuntime.sync();
  }
}
