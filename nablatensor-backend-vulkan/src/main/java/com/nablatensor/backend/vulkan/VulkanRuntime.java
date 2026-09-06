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

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Minimal headless Vulkan compute runtime via {@link java.lang.foreign}: one
 * instance / logical device / compute queue, GLSL compute shaders compiled to
 * SPIR-V in-process with {@code libshaderc}, dispatched one command buffer at a
 * time with {@code vkQueueWaitIdle} between submits. No JNI, no native jars.
 *
 * <p>Buffers are placed in {@code HOST_VISIBLE | HOST_COHERENT} memory (an APU
 * exposes this as device-local too) and persistently mapped at allocation, so
 * upload/download are a bare {@code memcpy} with no staging buffers and no
 * per-call map/unmap.
 */
final class VulkanRuntime {

  private VulkanRuntime() {
  }

  private static final Arena LIB = Arena.ofShared();
  private static final Linker LINKER = Linker.nativeLinker();

  // ---- symbol loading --------------------------------------------------

  private static final SymbolLookup VK = lookup("VULKAN_LOADER_PATH",
      new String[] {"/lib/x86_64-linux-gnu/libvulkan.so.1", "libvulkan.so.1", "libvulkan.so"});
  private static final SymbolLookup SHADERC = lookup("SHADERC_PATH",
      new String[] {"/lib/x86_64-linux-gnu/libshaderc.so.1", "libshaderc.so.1", "libshaderc.so"});

  private static SymbolLookup lookup(String env, String[] candidates) {
    String explicit = System.getenv(env);
    if (explicit != null && !explicit.isBlank()) {
      return SymbolLookup.libraryLookup(explicit, LIB);
    }
    for (String candidate : candidates) {
      try {
        return SymbolLookup.libraryLookup(candidate, LIB);
      } catch (RuntimeException ignored) {
        // try the next candidate
      }
    }
    return SymbolLookup.libraryLookup(candidates[candidates.length - 1], LIB);
  }

  private static MethodHandle vk(String name, FunctionDescriptor descriptor) {
    return LINKER.downcallHandle(VK.find(name).orElseThrow(() -> new UnsatisfiedLinkError(name)), descriptor);
  }

  private static MethodHandle sc(String name, FunctionDescriptor descriptor) {
    return LINKER.downcallHandle(SHADERC.find(name).orElseThrow(() -> new UnsatisfiedLinkError(name)), descriptor);
  }

  // result-returning entry points
  private static final MethodHandle CREATE_INSTANCE = vk("vkCreateInstance", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle ENUM_PHYS = vk("vkEnumeratePhysicalDevices", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));
  private static final MethodHandle PHYS_PROPS = vk("vkGetPhysicalDeviceProperties", FunctionDescriptor.ofVoid(JAVA_LONG, ADDRESS));
  private static final MethodHandle QFAM_PROPS = vk("vkGetPhysicalDeviceQueueFamilyProperties", FunctionDescriptor.ofVoid(JAVA_LONG, ADDRESS, ADDRESS));
  private static final MethodHandle MEM_PROPS = vk("vkGetPhysicalDeviceMemoryProperties", FunctionDescriptor.ofVoid(JAVA_LONG, ADDRESS));
  private static final MethodHandle CREATE_DEVICE = vk("vkCreateDevice", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle GET_QUEUE = vk("vkGetDeviceQueue", FunctionDescriptor.ofVoid(JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS));
  private static final MethodHandle CREATE_BUFFER = vk("vkCreateBuffer", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle DESTROY_BUFFER = vk("vkDestroyBuffer", FunctionDescriptor.ofVoid(JAVA_LONG, JAVA_LONG, ADDRESS));
  private static final MethodHandle BUFFER_MEM_REQ = vk("vkGetBufferMemoryRequirements", FunctionDescriptor.ofVoid(JAVA_LONG, JAVA_LONG, ADDRESS));
  private static final MethodHandle ALLOC_MEM = vk("vkAllocateMemory", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle FREE_MEM = vk("vkFreeMemory", FunctionDescriptor.ofVoid(JAVA_LONG, JAVA_LONG, ADDRESS));
  private static final MethodHandle BIND_BUFFER_MEM = vk("vkBindBufferMemory", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG));
  private static final MethodHandle MAP_MEM = vk("vkMapMemory", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS));
  private static final MethodHandle UNMAP_MEM = vk("vkUnmapMemory", FunctionDescriptor.ofVoid(JAVA_LONG, JAVA_LONG));
  private static final MethodHandle CREATE_SHADER = vk("vkCreateShaderModule", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_DSL = vk("vkCreateDescriptorSetLayout", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_PL = vk("vkCreatePipelineLayout", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_COMPUTE_PIPE = vk("vkCreateComputePipelines", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_PIPELINE_CACHE = vk("vkCreatePipelineCache", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle GET_PIPELINE_CACHE_DATA = vk("vkGetPipelineCacheData", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_DPOOL = vk("vkCreateDescriptorPool", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle ALLOC_DSET = vk("vkAllocateDescriptorSets", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));
  private static final MethodHandle FREE_DSET = vk("vkFreeDescriptorSets", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS));
  private static final MethodHandle UPDATE_DSETS = vk("vkUpdateDescriptorSets", FunctionDescriptor.ofVoid(JAVA_LONG, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
  private static final MethodHandle CREATE_CPOOL = vk("vkCreateCommandPool", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle ALLOC_CMD = vk("vkAllocateCommandBuffers", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));
  private static final MethodHandle BEGIN_CMD = vk("vkBeginCommandBuffer", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS));
  private static final MethodHandle END_CMD = vk("vkEndCommandBuffer", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
  private static final MethodHandle RESET_CMD = vk("vkResetCommandBuffer", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT));
  private static final MethodHandle CMD_BIND_PIPE = vk("vkCmdBindPipeline", FunctionDescriptor.ofVoid(JAVA_LONG, JAVA_INT, JAVA_LONG));
  private static final MethodHandle CMD_BIND_DSETS = vk("vkCmdBindDescriptorSets", FunctionDescriptor.ofVoid(JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
  private static final MethodHandle CMD_PUSH = vk("vkCmdPushConstants", FunctionDescriptor.ofVoid(JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS));
  private static final MethodHandle CMD_DISPATCH = vk("vkCmdDispatch", FunctionDescriptor.ofVoid(JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));
  private static final MethodHandle CMD_BARRIER = vk("vkCmdPipelineBarrier",
      FunctionDescriptor.ofVoid(JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
  private static final MethodHandle QUEUE_SUBMIT = vk("vkQueueSubmit", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG));
  private static final MethodHandle QUEUE_WAIT = vk("vkQueueWaitIdle", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));

  private static final MethodHandle SC_INIT = sc("shaderc_compiler_initialize", FunctionDescriptor.of(ADDRESS));
  private static final MethodHandle SC_COMPILE = sc("shaderc_compile_into_spv",
      FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle SC_STATUS = sc("shaderc_result_get_compilation_status", FunctionDescriptor.of(JAVA_INT, ADDRESS));
  private static final MethodHandle SC_LEN = sc("shaderc_result_get_length", FunctionDescriptor.of(JAVA_LONG, ADDRESS));
  private static final MethodHandle SC_BYTES = sc("shaderc_result_get_bytes", FunctionDescriptor.of(ADDRESS, ADDRESS));
  private static final MethodHandle SC_ERR = sc("shaderc_result_get_error_message", FunctionDescriptor.of(ADDRESS, ADDRESS));
  private static final MethodHandle SC_RELEASE = sc("shaderc_result_release", FunctionDescriptor.ofVoid(ADDRESS));

  // ---- Vulkan constants ------------------------------------------------

  private static final int ST_INSTANCE = 1;
  private static final int ST_DEVICE_QUEUE = 2;
  private static final int ST_DEVICE = 3;
  private static final int ST_SUBMIT = 4;
  private static final int ST_MEM_ALLOC = 5;
  private static final int ST_BUFFER = 12;
  private static final int ST_SHADER_MODULE = 16;
  private static final int ST_PIPELINE_CACHE = 17;
  private static final int ST_SHADER_STAGE = 18;
  private static final int ST_COMPUTE_PIPELINE = 29;
  private static final int ST_PIPELINE_LAYOUT = 30;
  private static final int ST_DSL = 32;
  private static final int ST_DPOOL = 33;
  private static final int ST_DSET_ALLOC = 34;
  private static final int ST_WRITE_DSET = 35;
  private static final int ST_CPOOL = 39;
  private static final int ST_CMD_ALLOC = 40;
  private static final int ST_CMD_BEGIN = 42;
  private static final int ST_MEMORY_BARRIER = 46;

  private static final int STAGE_COMPUTE = 0x20;
  private static final int STAGE_COMPUTE_SHADER = 0x800;
  private static final int ACCESS_SHADER_READ = 0x20;
  private static final int ACCESS_SHADER_WRITE = 0x40;
  /** Recorded dispatches before an automatic flush (bounds descriptor-pool + latency). */
  private static final int BATCH_CAP = 128;
  private static final int DTYPE_STORAGE = 7;
  private static final int USAGE_STORAGE_XFER = 0x20 | 0x1 | 0x2;
  private static final int MEM_DEVICE_LOCAL = 1;
  private static final int MEM_HOST_VISIBLE = 2;
  private static final int MEM_HOST_COHERENT = 4;
  private static final int QUEUE_COMPUTE = 2;
  private static final int BIND_POINT_COMPUTE = 1;
  private static final int CPOOL_RESET_BIT = 2;
  private static final int DPOOL_FREE_BIT = 1;
  private static final int CMD_ONE_TIME = 1;
  private static final long WHOLE_SIZE = -1L;
  private static final int PUSH_BYTES = 128;

  private static void check(int status, String op) {
    if (status != 0) {
      throw new RuntimeException(op + " failed: VkResult " + status);
    }
  }

  // ---- device state -------------------------------------------------------

  private static boolean ready;
  private static long instance;
  private static long physical;
  private static long device;
  private static long queue;
  private static long commandPool;
  private static long commandBuffer;
  private static long descriptorPool;
  private static long pipelineCache;
  private static int memoryTypeIndex = -1;
  private static String deviceName = "";
  private static final Arena RT = Arena.ofShared();
  private static final Map<String, Pipeline> PIPELINES = new HashMap<>();
  /**
   * Persistent host mapping per device-memory handle. Every buffer this runtime
   * allocates lives in host-visible coherent memory on the shared-memory APU it
   * targets, so it is mapped once at {@link #alloc} and stays mapped for the
   * buffer's life; {@code writeFloats}/{@code readFloats} then reduce to a bare
   * {@code memcpy} instead of a {@code vkMapMemory}+{@code vkUnmapMemory} round
   * trip per call. That round trip is a measurable fraction of a small dispatch,
   * so this is real wall-clock on the many-tiny-dispatch workloads — engine
   * calibration, scenario ladders, per-node tensor readbacks.
   */
  private static final Map<Long, MemorySegment> MAPPED = new HashMap<>();
  private static long shadercCompiler;

  // ---- deferred-submit batch state (gap #2) ----
  /** True while {@link #commandBuffer} is open and accumulating dispatches not yet submitted. */
  private static boolean recording;
  private static int pendingDispatches;
  /** Descriptor sets bound by recorded-but-unsubmitted dispatches; freed once the batch retires. */
  private static final List<Long> pendingSets = new ArrayList<>();
  /** {@code [buffer, memory]} pairs whose free was requested during a batch; run after it retires. */
  private static final List<long[]> pendingFrees = new ArrayList<>();

  private record Pipeline(long handle, long layout, long descriptorLayout, int bindings) {
  }

  static boolean probe() {
    try {
      init();
      return ready;
    } catch (Throwable failure) {
      return false;
    }
  }

  static String deviceName() {
    init();
    return deviceName;
  }

  static synchronized void init() {
    if (ready) {
      return;
    }
    try (Arena a = Arena.ofConfined()) {
      // instance
      MemorySegment ici = a.allocate(64);
      ici.set(JAVA_INT, 0, ST_INSTANCE);
      MemorySegment instOut = a.allocate(JAVA_LONG);
      check((int) CREATE_INSTANCE.invoke(ici, MemorySegment.NULL, instOut), "vkCreateInstance");
      instance = instOut.get(JAVA_LONG, 0);

      // physical device: prefer a real GPU over a CPU implementation
      MemorySegment countSeg = a.allocate(JAVA_INT);
      check((int) ENUM_PHYS.invoke(instance, countSeg, MemorySegment.NULL), "vkEnumeratePhysicalDevices");
      int count = countSeg.get(JAVA_INT, 0);
      if (count == 0) {
        throw new RuntimeException("no Vulkan physical devices");
      }
      MemorySegment handles = a.allocate((long) count * JAVA_LONG.byteSize());
      check((int) ENUM_PHYS.invoke(instance, countSeg, handles), "vkEnumeratePhysicalDevices");
      int bestScore = Integer.MIN_VALUE;
      MemorySegment props = a.allocate(1024);
      for (int i = 0; i < count; i++) {
        long candidate = handles.getAtIndex(JAVA_LONG, i);
        PHYS_PROPS.invoke(candidate, props);
        int type = props.get(JAVA_INT, 16); // OTHER0 iGPU1 dGPU2 vGPU3 CPU4
        int score = switch (type) {
          case 2 -> 4;
          case 1 -> 3;
          case 3 -> 2;
          case 0 -> 1;
          default -> 0;
        };
        if (score > bestScore) {
          bestScore = score;
          physical = candidate;
          deviceName = props.reinterpret(20 + 256).getString(20);
        }
      }

      // compute queue family
      MemorySegment qCount = a.allocate(JAVA_INT);
      QFAM_PROPS.invoke(physical, qCount, MemorySegment.NULL);
      int families = qCount.get(JAVA_INT, 0);
      MemorySegment qProps = a.allocate((long) families * 24);
      QFAM_PROPS.invoke(physical, qCount, qProps);
      int computeFamily = -1;
      for (int i = 0; i < families; i++) {
        int flags = qProps.get(JAVA_INT, (long) i * 24);
        if ((flags & QUEUE_COMPUTE) != 0) {
          computeFamily = i;
          break;
        }
      }
      if (computeFamily < 0) {
        throw new RuntimeException("no Vulkan compute queue family");
      }

      // logical device + queue
      MemorySegment priority = a.allocate(JAVA_FLOAT);
      priority.set(JAVA_FLOAT, 0, 1.0f);
      MemorySegment dqci = a.allocate(40);
      dqci.set(JAVA_INT, 0, ST_DEVICE_QUEUE);
      dqci.set(JAVA_INT, 20, computeFamily);
      dqci.set(JAVA_INT, 24, 1);
      dqci.set(ADDRESS, 32, priority);
      MemorySegment dci = a.allocate(72);
      dci.set(JAVA_INT, 0, ST_DEVICE);
      dci.set(JAVA_INT, 20, 1);
      dci.set(ADDRESS, 24, dqci);
      MemorySegment devOut = a.allocate(JAVA_LONG);
      check((int) CREATE_DEVICE.invoke(physical, dci, MemorySegment.NULL, devOut), "vkCreateDevice");
      device = devOut.get(JAVA_LONG, 0);
      MemorySegment queueOut = a.allocate(JAVA_LONG);
      GET_QUEUE.invoke(device, computeFamily, 0, queueOut);
      queue = queueOut.get(JAVA_LONG, 0);

      // host-visible + coherent memory type (prefer device-local as well)
      MemorySegment memProps = a.allocate(1024);
      MEM_PROPS.invoke(physical, memProps);
      int typeCount = memProps.get(JAVA_INT, 0);
      int fallback = -1;
      for (int i = 0; i < typeCount; i++) {
        int flags = memProps.get(JAVA_INT, 4 + (long) i * 8);
        if ((flags & MEM_HOST_VISIBLE) != 0 && (flags & MEM_HOST_COHERENT) != 0) {
          if ((flags & MEM_DEVICE_LOCAL) != 0) {
            memoryTypeIndex = i;
            break;
          }
          if (fallback < 0) {
            fallback = i;
          }
        }
      }
      if (memoryTypeIndex < 0) {
        memoryTypeIndex = fallback;
      }
      if (memoryTypeIndex < 0) {
        throw new RuntimeException("no host-visible coherent Vulkan memory type");
      }

      // command pool + one primary command buffer
      MemorySegment cpci = a.allocate(24);
      cpci.set(JAVA_INT, 0, ST_CPOOL);
      cpci.set(JAVA_INT, 16, CPOOL_RESET_BIT);
      cpci.set(JAVA_INT, 20, computeFamily);
      MemorySegment cpOut = a.allocate(JAVA_LONG);
      check((int) CREATE_CPOOL.invoke(device, cpci, MemorySegment.NULL, cpOut), "vkCreateCommandPool");
      commandPool = cpOut.get(JAVA_LONG, 0);
      MemorySegment cbai = a.allocate(32);
      cbai.set(JAVA_INT, 0, ST_CMD_ALLOC);
      cbai.set(JAVA_LONG, 16, commandPool);
      cbai.set(JAVA_INT, 24, 0); // PRIMARY
      cbai.set(JAVA_INT, 28, 1);
      MemorySegment cbOut = a.allocate(JAVA_LONG);
      check((int) ALLOC_CMD.invoke(device, cbai, cbOut), "vkAllocateCommandBuffers");
      commandBuffer = cbOut.get(JAVA_LONG, 0);

      // descriptor pool
      MemorySegment poolSize = a.allocate(8);
      poolSize.set(JAVA_INT, 0, DTYPE_STORAGE);
      poolSize.set(JAVA_INT, 4, 8192);
      MemorySegment dpci = a.allocate(40);
      dpci.set(JAVA_INT, 0, ST_DPOOL);
      dpci.set(JAVA_INT, 16, DPOOL_FREE_BIT);
      dpci.set(JAVA_INT, 20, 4096); // maxSets
      dpci.set(JAVA_INT, 24, 1);
      dpci.set(ADDRESS, 32, poolSize);
      MemorySegment dpOut = a.allocate(JAVA_LONG);
      check((int) CREATE_DPOOL.invoke(device, dpci, MemorySegment.NULL, dpOut), "vkCreateDescriptorPool");
      descriptorPool = dpOut.get(JAVA_LONG, 0);

      // pipeline cache, seeded from disk so a warm run skips shader compilation
      byte[] seed = readPipelineCacheFile();
      MemorySegment pcci = a.allocate(40);
      pcci.set(JAVA_INT, 0, ST_PIPELINE_CACHE);
      if (seed.length > 0) {
        MemorySegment seedSeg = RT.allocate(seed.length);
        MemorySegment.copy(seed, 0, seedSeg, JAVA_BYTE, 0, seed.length);
        pcci.set(JAVA_LONG, 24, seed.length);
        pcci.set(ADDRESS, 32, seedSeg);
      }
      MemorySegment pcOut = a.allocate(JAVA_LONG);
      if ((int) CREATE_PIPELINE_CACHE.invoke(device, pcci, MemorySegment.NULL, pcOut) == 0) {
        pipelineCache = pcOut.get(JAVA_LONG, 0);
      }

      shadercCompiler = ((MemorySegment) SC_INIT.invoke()).address();
      ready = true;
      Runtime.getRuntime().addShutdownHook(new Thread(VulkanRuntime::savePipelineCache, "nablatensor-vk-pipeline-cache"));
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static Path pipelineCacheFile() {
    String override = System.getProperty("nablatensor.tensor.vulkan.pipeline_cache");
    if (override != null && !override.isBlank()) {
      return Path.of(override);
    }
    String xdg = System.getenv("XDG_CACHE_HOME");
    Path base = (xdg != null && !xdg.isBlank())
        ? Path.of(xdg)
        : Path.of(System.getProperty("user.home", "."), ".cache");
    return base.resolve("nablatensor").resolve("vk-pipeline-cache.bin");
  }

  private static byte[] readPipelineCacheFile() {
    try {
      Path file = pipelineCacheFile();
      return Files.isRegularFile(file) ? Files.readAllBytes(file) : new byte[0];
    } catch (Exception ignored) {
      return new byte[0];
    }
  }

  /** Persists the driver's pipeline cache so the next process starts warm. Best effort. */
  static void savePipelineCache() {
    if (!ready || pipelineCache == 0) {
      return;
    }
    try (Arena a = Arena.ofConfined()) {
      MemorySegment sizePtr = a.allocate(JAVA_LONG);
      if ((int) GET_PIPELINE_CACHE_DATA.invoke(device, pipelineCache, sizePtr, MemorySegment.NULL) != 0) {
        return;
      }
      long size = sizePtr.get(JAVA_LONG, 0);
      if (size <= 0) {
        return;
      }
      MemorySegment data = a.allocate(size);
      if ((int) GET_PIPELINE_CACHE_DATA.invoke(device, pipelineCache, sizePtr, data) != 0) {
        return;
      }
      long written = sizePtr.get(JAVA_LONG, 0);
      byte[] bytes = new byte[Math.toIntExact(written)];
      MemorySegment.copy(data, JAVA_BYTE, 0, bytes, 0, bytes.length);
      Path file = pipelineCacheFile();
      Files.createDirectories(file.getParent());
      Files.write(file, bytes);
    } catch (Throwable ignored) {
      // a missing cache only costs shader-compile time next start
    }
  }

  // ---- shader compilation --------------------------------------------------

  static byte[] compileGlsl(String source, String name) {
    init();
    try (Arena a = Arena.ofConfined()) {
      MemorySegment src = a.allocateFrom(source);
      MemorySegment file = a.allocateFrom(name);
      MemorySegment entry = a.allocateFrom("main");
      MemorySegment result = (MemorySegment) SC_COMPILE.invoke(
          MemorySegment.ofAddress(shadercCompiler), src, source.getBytes(java.nio.charset.StandardCharsets.UTF_8).length,
          2 /* shaderc_compute_shader */, file, entry, MemorySegment.NULL);
      try {
        int status = (int) SC_STATUS.invoke(result);
        if (status != 0) {
          MemorySegment err = (MemorySegment) SC_ERR.invoke(result);
          throw new RuntimeException("shaderc failed for " + name + " (status " + status + "):\n"
              + err.reinterpret(4096).getString(0) + "\n--- source ---\n" + source);
        }
        long length = (long) SC_LEN.invoke(result);
        MemorySegment bytes = ((MemorySegment) SC_BYTES.invoke(result)).reinterpret(length);
        byte[] spirv = new byte[Math.toIntExact(length)];
        MemorySegment.copy(bytes, JAVA_BYTE, 0, spirv, 0, spirv.length);
        return spirv;
      } finally {
        SC_RELEASE.invoke(result);
      }
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static synchronized void registerPipeline(String name, String glsl, int bindings) {
    init();
    if (PIPELINES.containsKey(name)) {
      return;
    }
    byte[] spirv = compileGlsl(glsl, name);
    try (Arena a = Arena.ofConfined()) {
      // shader module
      MemorySegment code = RT.allocate(spirv.length);
      MemorySegment.copy(spirv, 0, code, JAVA_BYTE, 0, spirv.length);
      MemorySegment smci = a.allocate(40);
      smci.set(JAVA_INT, 0, ST_SHADER_MODULE);
      smci.set(JAVA_LONG, 24, spirv.length);
      smci.set(ADDRESS, 32, code);
      MemorySegment smOut = a.allocate(JAVA_LONG);
      check((int) CREATE_SHADER.invoke(device, smci, MemorySegment.NULL, smOut), "vkCreateShaderModule");
      long module = smOut.get(JAVA_LONG, 0);

      // descriptor set layout: `bindings` storage buffers, all compute stage
      MemorySegment bindingArray = a.allocate((long) bindings * 24);
      for (int i = 0; i < bindings; i++) {
        long base = (long) i * 24;
        bindingArray.set(JAVA_INT, base, i);
        bindingArray.set(JAVA_INT, base + 4, DTYPE_STORAGE);
        bindingArray.set(JAVA_INT, base + 8, 1);
        bindingArray.set(JAVA_INT, base + 12, STAGE_COMPUTE);
      }
      MemorySegment dslci = a.allocate(32);
      dslci.set(JAVA_INT, 0, ST_DSL);
      dslci.set(JAVA_INT, 20, bindings);
      dslci.set(ADDRESS, 24, bindingArray);
      MemorySegment dslOut = a.allocate(JAVA_LONG);
      check((int) CREATE_DSL.invoke(device, dslci, MemorySegment.NULL, dslOut), "vkCreateDescriptorSetLayout");
      long dsl = dslOut.get(JAVA_LONG, 0);

      // pipeline layout with a 128-byte compute push-constant range
      MemorySegment pcr = a.allocate(12);
      pcr.set(JAVA_INT, 0, STAGE_COMPUTE);
      pcr.set(JAVA_INT, 4, 0);
      pcr.set(JAVA_INT, 8, PUSH_BYTES);
      MemorySegment dslHandle = a.allocate(JAVA_LONG);
      dslHandle.set(JAVA_LONG, 0, dsl);
      MemorySegment plci = a.allocate(48);
      plci.set(JAVA_INT, 0, ST_PIPELINE_LAYOUT);
      plci.set(JAVA_INT, 20, 1);
      plci.set(ADDRESS, 24, dslHandle);
      plci.set(JAVA_INT, 32, 1);
      plci.set(ADDRESS, 40, pcr);
      MemorySegment plOut = a.allocate(JAVA_LONG);
      check((int) CREATE_PL.invoke(device, plci, MemorySegment.NULL, plOut), "vkCreatePipelineLayout");
      long layout = plOut.get(JAVA_LONG, 0);

      // compute pipeline (stage struct embedded at offset 24)
      MemorySegment entryName = RT.allocateFrom("main");
      MemorySegment cpci = a.allocate(96);
      cpci.set(JAVA_INT, 0, ST_COMPUTE_PIPELINE);
      cpci.set(JAVA_INT, 24, ST_SHADER_STAGE);
      cpci.set(JAVA_INT, 24 + 20, STAGE_COMPUTE);
      cpci.set(JAVA_LONG, 24 + 24, module);
      cpci.set(ADDRESS, 24 + 32, entryName);
      cpci.set(JAVA_LONG, 72, layout);
      cpci.set(JAVA_INT, 88, -1); // basePipelineIndex
      MemorySegment pipeOut = a.allocate(JAVA_LONG);
      check((int) CREATE_COMPUTE_PIPE.invoke(device, pipelineCache, 1, cpci, MemorySegment.NULL, pipeOut), "vkCreateComputePipelines");
      long pipeline = pipeOut.get(JAVA_LONG, 0);

      PIPELINES.put(name, new Pipeline(pipeline, layout, dsl, bindings));
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  // ---- buffers ----------------------------------------------------------

  /** {@code [buffer, memory]} handles. */
  static long[] alloc(long bytes) {
    init();
    try (Arena a = Arena.ofConfined()) {
      MemorySegment bci = a.allocate(64);
      bci.set(JAVA_INT, 0, ST_BUFFER);
      bci.set(JAVA_LONG, 24, bytes);
      bci.set(JAVA_INT, 32, USAGE_STORAGE_XFER);
      MemorySegment bufOut = a.allocate(JAVA_LONG);
      check((int) CREATE_BUFFER.invoke(device, bci, MemorySegment.NULL, bufOut), "vkCreateBuffer");
      long buffer = bufOut.get(JAVA_LONG, 0);

      MemorySegment req = a.allocate(24);
      BUFFER_MEM_REQ.invoke(device, buffer, req);
      long size = req.get(JAVA_LONG, 0);

      MemorySegment mai = a.allocate(32);
      mai.set(JAVA_INT, 0, ST_MEM_ALLOC);
      mai.set(JAVA_LONG, 16, size);
      mai.set(JAVA_INT, 24, memoryTypeIndex);
      MemorySegment memOut = a.allocate(JAVA_LONG);
      check((int) ALLOC_MEM.invoke(device, mai, MemorySegment.NULL, memOut), "vkAllocateMemory");
      long memory = memOut.get(JAVA_LONG, 0);
      check((int) BIND_BUFFER_MEM.invoke(device, buffer, memory, 0L), "vkBindBufferMemory");

      // Map once, for the buffer's whole life (see MAPPED). Coherent memory, so
      // no explicit flush/invalidate is ever needed around the later copies.
      MemorySegment mapPtr = a.allocate(ADDRESS);
      check((int) MAP_MEM.invoke(device, memory, 0L, WHOLE_SIZE, 0, mapPtr), "vkMapMemory");
      MAPPED.put(memory, mapPtr.get(ADDRESS, 0).reinterpret(size));
      return new long[] {buffer, memory};
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static synchronized void free(long buffer, long memory) {
    if (recording || pendingDispatches > 0) {
      // a recorded-but-unsubmitted dispatch may still reference this buffer
      pendingFrees.add(new long[] {buffer, memory});
      return;
    }
    try {
      if (MAPPED.remove(memory) != null) {
        UNMAP_MEM.invoke(device, memory);
      }
      DESTROY_BUFFER.invoke(device, buffer, MemorySegment.NULL);
      FREE_MEM.invoke(device, memory, MemorySegment.NULL);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static synchronized void writeFloats(long memory, float[] data) {
    flushLocked();   // a pending dispatch may read this buffer
    try {
      MemorySegment mapped = mappedFor(memory, (long) data.length * Float.BYTES);
      MemorySegment.copy(data, 0, mapped, JAVA_FLOAT, 0, data.length);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static synchronized float[] readFloats(long memory, int count) {
    flushLocked();   // the data we want may still be a pending dispatch's output
    float[] out = new float[count];
    try {
      MemorySegment mapped = mappedFor(memory, (long) count * Float.BYTES);
      MemorySegment.copy(mapped, JAVA_FLOAT, 0, out, 0, count);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
    return out;
  }

  /**
   * The persistent host mapping for {@code memory}, sized to at least
   * {@code bytes}. Falls back to a one-shot {@code vkMapMemory} for any buffer
   * this runtime did not allocate (there are none today, but the map stays
   * total).
   */
  private static MemorySegment mappedFor(long memory, long bytes) throws Throwable {
    MemorySegment mapped = MAPPED.get(memory);
    if (mapped != null) {
      return mapped.byteSize() >= bytes ? mapped : mapped.reinterpret(bytes);
    }
    try (Arena a = Arena.ofConfined()) {
      MemorySegment pp = a.allocate(ADDRESS);
      check((int) MAP_MEM.invoke(device, memory, 0L, WHOLE_SIZE, 0, pp), "vkMapMemory");
      MemorySegment fresh = pp.get(ADDRESS, 0).reinterpret(bytes);
      MAPPED.put(memory, fresh);
      return fresh;
    }
  }

  // ---- dispatch ----------------------------------------------------------

  private static final Arena PUSH_ARENA = Arena.ofAuto();
  private static final MemorySegment PUSH = PUSH_ARENA.allocate(PUSH_BYTES);

  /**
   * <b>Records</b> one dispatch of {@code kernel} into the open batch command
   * buffer (opening it first if needed) and a compute→compute memory barrier
   * after it, but does <b>not</b> submit. Work is submitted — once for the whole
   * batch — by {@link #flush()} / {@link #sync()} / a host {@code readFloats} /
   * {@code writeFloats}, or automatically after {@value #BATCH_CAP} recorded
   * dispatches. Descriptor sets and any {@link #free} requested meanwhile are
   * held until the batch retires, so a caller may {@code release()} an
   * intermediate immediately after issuing the op that consumes it.
   */
  static synchronized void dispatch(String kernel, int groupsX, int groupsY, int groupsZ,
      long[] buffers, int[] pushInts, int pushFloatSlot, float pushFloat) {
    Pipeline pipeline = PIPELINES.get(kernel);
    if (pipeline == null) {
      throw new IllegalStateException("unregistered Vulkan kernel: " + kernel);
    }
    try (Arena a = Arena.ofConfined()) {
      ensureRecording(a);

      // descriptor set (freed only once the batch retires)
      MemorySegment dslHandle = a.allocate(JAVA_LONG);
      dslHandle.set(JAVA_LONG, 0, pipeline.descriptorLayout());
      MemorySegment dsai = a.allocate(40);
      dsai.set(JAVA_INT, 0, ST_DSET_ALLOC);
      dsai.set(JAVA_LONG, 16, descriptorPool);
      dsai.set(JAVA_INT, 24, 1);
      dsai.set(ADDRESS, 32, dslHandle);
      MemorySegment dsOut = a.allocate(JAVA_LONG);
      check((int) ALLOC_DSET.invoke(device, dsai, dsOut), "vkAllocateDescriptorSets");
      long descriptorSet = dsOut.get(JAVA_LONG, 0);

      int n = pipeline.bindings();
      MemorySegment bufferInfos = a.allocate((long) n * 24);
      MemorySegment writes = a.allocate((long) n * 64);
      for (int i = 0; i < n; i++) {
        long bi = (long) i * 24;
        bufferInfos.set(JAVA_LONG, bi, buffers[i]);
        bufferInfos.set(JAVA_LONG, bi + 8, 0L);
        bufferInfos.set(JAVA_LONG, bi + 16, WHOLE_SIZE);
        long wi = (long) i * 64;
        writes.set(JAVA_INT, wi, ST_WRITE_DSET);
        writes.set(JAVA_LONG, wi + 16, descriptorSet);
        writes.set(JAVA_INT, wi + 24, i);       // dstBinding
        writes.set(JAVA_INT, wi + 32, 1);       // descriptorCount
        writes.set(JAVA_INT, wi + 36, DTYPE_STORAGE);
        writes.set(ADDRESS, wi + 48, bufferInfos.asSlice(bi, 24));
      }
      UPDATE_DSETS.invoke(device, n, writes, 0, MemorySegment.NULL);
      pendingSets.add(descriptorSet);

      // push constants — vkCmdPushConstants copies PUSH into the command buffer here
      PUSH.fill((byte) 0);
      if (pushInts != null) {
        for (int i = 0; i < pushInts.length; i++) {
          PUSH.set(JAVA_INT, (long) i * 4, pushInts[i]);
        }
      }
      if (pushFloatSlot >= 0) {
        PUSH.set(JAVA_FLOAT, (long) pushFloatSlot * 4, pushFloat);
      }

      MemorySegment setHandle = a.allocate(JAVA_LONG);
      setHandle.set(JAVA_LONG, 0, descriptorSet);

      CMD_BIND_PIPE.invoke(commandBuffer, BIND_POINT_COMPUTE, pipeline.handle());
      CMD_BIND_DSETS.invoke(commandBuffer, BIND_POINT_COMPUTE, pipeline.layout(), 0, 1, setHandle, 0, MemorySegment.NULL);
      CMD_PUSH.invoke(commandBuffer, pipeline.layout(), STAGE_COMPUTE, 0, PUSH_BYTES, PUSH);
      CMD_DISPATCH.invoke(commandBuffer, groupsX, groupsY, groupsZ);

      // conservative global barrier: the next recorded dispatch sees these writes
      MemorySegment barrier = a.allocate(24);
      barrier.set(JAVA_INT, 0, ST_MEMORY_BARRIER);
      barrier.set(JAVA_INT, 16, ACCESS_SHADER_WRITE);
      barrier.set(JAVA_INT, 20, ACCESS_SHADER_READ | ACCESS_SHADER_WRITE);
      CMD_BARRIER.invoke(commandBuffer, STAGE_COMPUTE_SHADER, STAGE_COMPUTE_SHADER, 0,
          1, barrier, 0, MemorySegment.NULL, 0, MemorySegment.NULL);

      if (++pendingDispatches >= BATCH_CAP) {
        flushLocked();
      }
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static void ensureRecording(Arena a) throws Throwable {
    if (recording) {
      return;
    }
    RESET_CMD.invoke(commandBuffer, 0);
    MemorySegment begin = a.allocate(32);
    begin.set(JAVA_INT, 0, ST_CMD_BEGIN);
    begin.set(JAVA_INT, 16, CMD_ONE_TIME);
    check((int) BEGIN_CMD.invoke(commandBuffer, begin), "vkBeginCommandBuffer");
    recording = true;
  }

  /** Submits the open batch (if any), waits for it, then releases its descriptor sets and deferred frees. */
  static synchronized void flush() {
    flushLocked();
  }

  private static void flushLocked() {
    try (Arena a = Arena.ofConfined()) {
      if (recording) {
        check((int) END_CMD.invoke(commandBuffer), "vkEndCommandBuffer");
        MemorySegment cmdHandle = a.allocate(JAVA_LONG);
        cmdHandle.set(JAVA_LONG, 0, commandBuffer);
        MemorySegment submit = a.allocate(72);
        submit.set(JAVA_INT, 0, ST_SUBMIT);
        submit.set(JAVA_INT, 40, 1);
        submit.set(ADDRESS, 48, cmdHandle);
        check((int) QUEUE_SUBMIT.invoke(queue, 1, submit, 0L), "vkQueueSubmit");
        check((int) QUEUE_WAIT.invoke(queue), "vkQueueWaitIdle");
        recording = false;
        pendingDispatches = 0;
      }
      if (!pendingSets.isEmpty()) {
        int count = pendingSets.size();
        MemorySegment sets = a.allocate((long) count * JAVA_LONG.byteSize());
        for (int i = 0; i < count; i++) {
          sets.setAtIndex(JAVA_LONG, i, pendingSets.get(i));
        }
        FREE_DSET.invoke(device, descriptorPool, count, sets);
        pendingSets.clear();
      }
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
    processPendingFrees();
  }

  private static void processPendingFrees() {
    if (pendingFrees.isEmpty()) {
      return;
    }
    for (long[] pair : pendingFrees) {
      try {
        if (MAPPED.remove(pair[1]) != null) {
          UNMAP_MEM.invoke(device, pair[1]);
        }
        DESTROY_BUFFER.invoke(device, pair[0], MemorySegment.NULL);
        FREE_MEM.invoke(device, pair[1], MemorySegment.NULL);
      } catch (Throwable ignored) {
        // teardown races are not fatal
      }
    }
    pendingFrees.clear();
  }

  static synchronized void sync() {
    if (!ready) {
      return;
    }
    flushLocked();
    try {
      check((int) QUEUE_WAIT.invoke(queue), "vkQueueWaitIdle");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }
}
