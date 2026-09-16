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

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.BooleanSupplier;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Minimal headless WebGPU compute runtime via {@link java.lang.foreign}, bound
 * to {@code wgpu-native}'s C ABI ({@code webgpu.h} / {@code wgpu.h}): one
 * instance / adapter / device / queue, WGSL compute shaders built directly
 * from source (no separate compile step, unlike Vulkan's GLSL→SPIR-V via
 * {@code libshaderc}), one command encoder + compute pass per dispatch.
 *
 * <p>This is the WebGPU counterpart of {@code VulkanRuntime}: same "no JNI, no
 * native jar" shape, same one-scenario-per-invocation replay kernel target.
 * Two real API differences shape it:
 *
 * <ul>
 *   <li>WebGPU has no push constants in its cross-platform surface, so the
 *       128-byte push-constant block Vulkan binds becomes a small read-only
 *       storage buffer at the kernel's last binding, updated with
 *       {@code wgpuQueueWriteBuffer} before each dispatch.</li>
 *   <li>A WebGPU storage buffer cannot be persistently host-mapped while the
 *       GPU may write it (unlike the host-visible/coherent memory an APU
 *       exposes to Vulkan), so host readback goes through the standard
 *       WebGPU staging path: copy device buffer → a dedicated
 *       {@code MapRead} buffer → {@code mapAsync} → read → unmap. Each
 *       {@link #dispatch} and the copy step in {@link #readFloats} therefore
 *       submit-and-wait individually rather than batching, unlike Vulkan's
 *       deferred multi-dispatch command buffer; the one caller here
 *       ({@code WgpuAadKernel}) always reads back straight after a dispatch
 *       anyway, so nothing is left on the table by keeping it simple.</li>
 * </ul>
 */
final class WgpuRuntime {

  private WgpuRuntime() {
  }

  private static final Arena LIB = Arena.ofShared();
  private static final Linker LINKER = Linker.nativeLinker();

  // ---- symbol loading --------------------------------------------------

  private static final SymbolLookup WGPU = lookup();

  private static SymbolLookup lookup() {
    String explicit = System.getenv("NABLATENSOR_WGPU_NATIVE_PATH");
    if (explicit == null || explicit.isBlank()) {
      explicit = System.getProperty("nablatensor.wgpu.native_path");
    }
    if (explicit != null && !explicit.isBlank()) {
      return SymbolLookup.libraryLookup(explicit, LIB);
    }
    String xdg = System.getenv("XDG_CACHE_HOME");
    Path cacheBase = (xdg != null && !xdg.isBlank())
        ? Path.of(xdg)
        : Path.of(System.getProperty("user.home", "."), ".cache");
    String cached = cacheBase.resolve("nablatensor").resolve("wgpu-native").resolve("lib")
        .resolve("libwgpu_native.so").toString();
    String[] candidates = {cached, "/usr/local/lib/libwgpu_native.so", "libwgpu_native.so"};
    for (String candidate : candidates) {
      try {
        return SymbolLookup.libraryLookup(candidate, LIB);
      } catch (RuntimeException ignored) {
        // try the next candidate
      }
    }
    return SymbolLookup.libraryLookup(candidates[candidates.length - 1], LIB);
  }

  private static MethodHandle wg(String name, FunctionDescriptor descriptor) {
    return LINKER.downcallHandle(WGPU.find(name).orElseThrow(() -> new UnsatisfiedLinkError(name)), descriptor);
  }

  // ---- by-value struct shapes (see webgpu.h) ----------------------------

  /**
   * {@code WGPUStringView}: {@code {char const * data; size_t length;}} — 16
   * bytes, both eightbytes classified INTEGER, so this coarse layout is ABI-
   * identical to the named struct for the by-value callback parameter it
   * appears in.
   */
  private static final MemoryLayout STRING_VIEW = MemoryLayout.structLayout(ADDRESS, JAVA_LONG);

  /**
   * The shape shared by {@code WGPURequestAdapterCallbackInfo},
   * {@code WGPURequestDeviceCallbackInfo} and {@code WGPUBufferMapCallbackInfo}:
   * {@code {WGPUChainedStruct* next; WGPUCallbackMode mode; <pad4>; callback;
   * userdata1; userdata2;}} — 40 bytes, passed by value (SysV MEMORY class,
   * &gt;16 bytes), so only total size/alignment matter to the calling
   * convention.
   */
  private static final MemoryLayout CALLBACK_INFO = MemoryLayout.structLayout(
      ADDRESS, JAVA_INT, MemoryLayout.paddingLayout(4), ADDRESS, ADDRESS, ADDRESS);

  // ---- enum / flag constants (see webgpu.h, wgpu.h) ---------------------

  private static final int CALLBACK_MODE_ALLOW_PROCESS_EVENTS = 0x2;
  private static final int STYPE_SHADER_SOURCE_WGSL = 0x2;
  private static final long SHADER_STAGE_COMPUTE = 0x4L;
  private static final int BUFFER_BINDING_TYPE_STORAGE = 0x3;
  private static final int BUFFER_BINDING_TYPE_READ_ONLY_STORAGE = 0x4;
  private static final long BUFFER_USAGE_MAP_READ = 0x1L;
  private static final long BUFFER_USAGE_COPY_SRC = 0x4L;
  private static final long BUFFER_USAGE_COPY_DST = 0x8L;
  private static final long BUFFER_USAGE_STORAGE = 0x80L;
  private static final long MAP_MODE_READ = 0x1L;
  private static final int REQUEST_ADAPTER_STATUS_SUCCESS = 1;
  private static final int REQUEST_DEVICE_STATUS_SUCCESS = 1;
  private static final int MAP_ASYNC_STATUS_SUCCESS = 1;
  private static final long WGPU_WHOLE_SIZE = -1L; // UINT64_MAX
  private static final long WGPU_STRLEN = -1L;      // UINT64_MAX: "use the null terminator"

  // ---- downcall handles ---------------------------------------------------

  private static final MethodHandle CREATE_INSTANCE = wg("wgpuCreateInstance", FunctionDescriptor.of(ADDRESS, ADDRESS));
  private static final MethodHandle REQUEST_ADAPTER = wg("wgpuInstanceRequestAdapter",
      FunctionDescriptor.of(JAVA_LONG, ADDRESS, ADDRESS, CALLBACK_INFO));
  private static final MethodHandle REQUEST_DEVICE = wg("wgpuAdapterRequestDevice",
      FunctionDescriptor.of(JAVA_LONG, ADDRESS, ADDRESS, CALLBACK_INFO));
  private static final MethodHandle PROCESS_EVENTS = wg("wgpuInstanceProcessEvents", FunctionDescriptor.ofVoid(ADDRESS));
  private static final MethodHandle ADAPTER_GET_INFO = wg("wgpuAdapterGetInfo", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle DEVICE_GET_QUEUE = wg("wgpuDeviceGetQueue", FunctionDescriptor.of(ADDRESS, ADDRESS));
  private static final MethodHandle DEVICE_POLL = wg("wgpuDevicePoll", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));

  private static final MethodHandle CREATE_BUFFER = wg("wgpuDeviceCreateBuffer", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle BUFFER_DESTROY = wg("wgpuBufferDestroy", FunctionDescriptor.ofVoid(ADDRESS));
  private static final MethodHandle BUFFER_RELEASE = wg("wgpuBufferRelease", FunctionDescriptor.ofVoid(ADDRESS));
  private static final MethodHandle BUFFER_MAP_ASYNC = wg("wgpuBufferMapAsync",
      FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, CALLBACK_INFO));
  private static final MethodHandle BUFFER_GET_CONST_MAPPED_RANGE = wg("wgpuBufferGetConstMappedRange",
      FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG));
  private static final MethodHandle BUFFER_UNMAP = wg("wgpuBufferUnmap", FunctionDescriptor.ofVoid(ADDRESS));

  private static final MethodHandle CREATE_SHADER_MODULE = wg("wgpuDeviceCreateShaderModule", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_BIND_GROUP_LAYOUT = wg("wgpuDeviceCreateBindGroupLayout", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_BIND_GROUP = wg("wgpuDeviceCreateBindGroup", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_PIPELINE_LAYOUT = wg("wgpuDeviceCreatePipelineLayout", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_COMPUTE_PIPELINE = wg("wgpuDeviceCreateComputePipeline", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle BIND_GROUP_RELEASE = wg("wgpuBindGroupRelease", FunctionDescriptor.ofVoid(ADDRESS));

  private static final MethodHandle CREATE_COMMAND_ENCODER = wg("wgpuDeviceCreateCommandEncoder", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle BEGIN_COMPUTE_PASS = wg("wgpuCommandEncoderBeginComputePass", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle COPY_BUFFER_TO_BUFFER = wg("wgpuCommandEncoderCopyBufferToBuffer",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_LONG));
  private static final MethodHandle COMMAND_ENCODER_FINISH = wg("wgpuCommandEncoderFinish", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle COMMAND_ENCODER_RELEASE = wg("wgpuCommandEncoderRelease", FunctionDescriptor.ofVoid(ADDRESS));
  private static final MethodHandle COMMAND_BUFFER_RELEASE = wg("wgpuCommandBufferRelease", FunctionDescriptor.ofVoid(ADDRESS));

  private static final MethodHandle COMPUTE_SET_PIPELINE = wg("wgpuComputePassEncoderSetPipeline", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
  private static final MethodHandle COMPUTE_SET_BIND_GROUP = wg("wgpuComputePassEncoderSetBindGroup",
      FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
  private static final MethodHandle COMPUTE_DISPATCH = wg("wgpuComputePassEncoderDispatchWorkgroups",
      FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT));
  private static final MethodHandle COMPUTE_PASS_END = wg("wgpuComputePassEncoderEnd", FunctionDescriptor.ofVoid(ADDRESS));
  private static final MethodHandle COMPUTE_PASS_RELEASE = wg("wgpuComputePassEncoderRelease", FunctionDescriptor.ofVoid(ADDRESS));

  private static final MethodHandle QUEUE_SUBMIT = wg("wgpuQueueSubmit", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, ADDRESS));
  private static final MethodHandle QUEUE_WRITE_BUFFER = wg("wgpuQueueWriteBuffer",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG));

  // ---- upcall stubs (native -> Java callbacks) --------------------------

  private static volatile boolean adapterDone;
  private static volatile int adapterStatus;
  private static volatile MemorySegment adapterHandle;

  private static volatile boolean deviceDone;
  private static volatile int deviceStatus;
  private static volatile MemorySegment deviceHandle;

  private static volatile boolean mapDone;
  private static volatile int mapStatus;

  @SuppressWarnings("unused") // invoked by native code via the upcall stub
  private static void onRequestAdapter(int status, MemorySegment adapter, MemorySegment message,
                                        MemorySegment ud1, MemorySegment ud2) {
    adapterStatus = status;
    adapterHandle = adapter;
    adapterDone = true;
  }

  @SuppressWarnings("unused")
  private static void onRequestDevice(int status, MemorySegment device, MemorySegment message,
                                       MemorySegment ud1, MemorySegment ud2) {
    deviceStatus = status;
    deviceHandle = device;
    deviceDone = true;
  }

  @SuppressWarnings("unused")
  private static void onBufferMap(int status, MemorySegment message, MemorySegment ud1, MemorySegment ud2) {
    mapStatus = status;
    mapDone = true;
  }

  private static final MemorySegment ADAPTER_CALLBACK_STUB = upcall("onRequestAdapter",
      MethodType.methodType(void.class, int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class, MemorySegment.class),
      FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, STRING_VIEW, ADDRESS, ADDRESS));
  private static final MemorySegment DEVICE_CALLBACK_STUB = upcall("onRequestDevice",
      MethodType.methodType(void.class, int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class, MemorySegment.class),
      FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, STRING_VIEW, ADDRESS, ADDRESS));
  private static final MemorySegment MAP_CALLBACK_STUB = upcall("onBufferMap",
      MethodType.methodType(void.class, int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class),
      FunctionDescriptor.ofVoid(JAVA_INT, STRING_VIEW, ADDRESS, ADDRESS));

  private static MemorySegment upcall(String methodName, MethodType type, FunctionDescriptor descriptor) {
    try {
      MethodHandle target = MethodHandles.lookup().findStatic(WgpuRuntime.class, methodName, type);
      return LINKER.upcallStub(target, descriptor, LIB);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  // ---- device state -------------------------------------------------------

  private static boolean ready;
  private static MemorySegment instance;
  private static MemorySegment adapter;
  private static MemorySegment device;
  private static MemorySegment queue;
  private static String deviceName = "";
  private static long readbackBuffer;
  private static long readbackCapacity;
  private static final Map<String, Pipeline> PIPELINES = new HashMap<>();

  private static final class Pipeline {
    final MemorySegment handle;
    final MemorySegment bindGroupLayout;
    private MemorySegment cachedBindGroup;
    private long[] cachedBuffers;

    Pipeline(MemorySegment handle, MemorySegment bindGroupLayout) {
      this.handle = handle;
      this.bindGroupLayout = bindGroupLayout;
    }

    synchronized MemorySegment bindGroupFor(long[] buffers) throws Throwable {
      if (cachedBindGroup != null && Arrays.equals(cachedBuffers, buffers)) {
        return cachedBindGroup;
      }
      if (cachedBindGroup != null) {
        BIND_GROUP_RELEASE.invoke(cachedBindGroup);
      }
      cachedBindGroup = buildBindGroup(bindGroupLayout, buffers);
      cachedBuffers = buffers.clone();
      return cachedBindGroup;
    }
  }

  private static MemorySegment ptr(long address) {
    return MemorySegment.ofAddress(address);
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
    try {
      instance = (MemorySegment) CREATE_INSTANCE.invoke(MemorySegment.NULL);
      if (instance.equals(MemorySegment.NULL)) {
        throw new RuntimeException("wgpuCreateInstance returned NULL");
      }
      requestAdapter();
      requestDevice();
      queue = (MemorySegment) DEVICE_GET_QUEUE.invoke(device);
      readDeviceName();
      ready = true;
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static void requestAdapter() throws Throwable {
    adapterDone = false;
    adapterStatus = 0;
    adapterHandle = null;
    try (Arena a = Arena.ofConfined()) {
      MemorySegment cb = a.allocate(CALLBACK_INFO);
      cb.set(ADDRESS, 0, MemorySegment.NULL);
      cb.set(JAVA_INT, 8, CALLBACK_MODE_ALLOW_PROCESS_EVENTS);
      cb.set(ADDRESS, 16, ADAPTER_CALLBACK_STUB);
      cb.set(ADDRESS, 24, MemorySegment.NULL);
      cb.set(ADDRESS, 32, MemorySegment.NULL);
      REQUEST_ADAPTER.invoke(instance, MemorySegment.NULL, cb);
    }
    waitFor(() -> adapterDone, "adapter request");
    if (adapterStatus != REQUEST_ADAPTER_STATUS_SUCCESS || adapterHandle == null || adapterHandle.equals(MemorySegment.NULL)) {
      throw new RuntimeException("no WebGPU adapter available (status " + adapterStatus + ")");
    }
    adapter = adapterHandle;
  }

  private static void requestDevice() throws Throwable {
    deviceDone = false;
    deviceStatus = 0;
    deviceHandle = null;
    try (Arena a = Arena.ofConfined()) {
      MemorySegment cb = a.allocate(CALLBACK_INFO);
      cb.set(ADDRESS, 0, MemorySegment.NULL);
      cb.set(JAVA_INT, 8, CALLBACK_MODE_ALLOW_PROCESS_EVENTS);
      cb.set(ADDRESS, 16, DEVICE_CALLBACK_STUB);
      cb.set(ADDRESS, 24, MemorySegment.NULL);
      cb.set(ADDRESS, 32, MemorySegment.NULL);
      REQUEST_DEVICE.invoke(adapter, MemorySegment.NULL, cb);
    }
    waitFor(() -> deviceDone, "device request");
    if (deviceStatus != REQUEST_DEVICE_STATUS_SUCCESS || deviceHandle == null || deviceHandle.equals(MemorySegment.NULL)) {
      throw new RuntimeException("no WebGPU device available (status " + deviceStatus + ")");
    }
    device = deviceHandle;
  }

  private static void waitFor(BooleanSupplier done, String what) throws Throwable {
    long deadline = System.nanoTime() + 5_000_000_000L;
    while (!done.getAsBoolean()) {
      PROCESS_EVENTS.invoke(instance);
      if (done.getAsBoolean()) {
        return;
      }
      if (System.nanoTime() > deadline) {
        throw new RuntimeException("timed out waiting for " + what);
      }
      Thread.sleep(1);
    }
  }

  private static void readDeviceName() {
    try (Arena a = Arena.ofConfined()) {
      MemorySegment info = a.allocate(96);
      int status = (int) ADAPTER_GET_INFO.invoke(adapter, info);
      if (status != 1) {
        deviceName = "wgpu device";
        return;
      }
      String dev = readStringView(info, 40);
      String desc = readStringView(info, 56);
      int backendType = info.get(JAVA_INT, 72);
      String backend = switch (backendType) {
        case 3 -> "D3D11";
        case 4 -> "D3D12";
        case 5 -> "Metal";
        case 6 -> "Vulkan";
        case 7 -> "OpenGL";
        case 8 -> "OpenGLES";
        default -> "backend " + backendType;
      };
      String name = !dev.isBlank() ? dev : !desc.isBlank() ? desc : "wgpu device";
      deviceName = name + " (" + backend + ")";
    } catch (Throwable failure) {
      deviceName = "wgpu device";
    }
  }

  /** Reads a {@code WGPUStringView} embedded at {@code offset} within {@code struct}. */
  private static String readStringView(MemorySegment struct, long offset) {
    MemorySegment data = struct.get(ADDRESS, offset);
    long length = struct.get(JAVA_LONG, offset + 8);
    if (data.equals(MemorySegment.NULL) || length <= 0) {
      return "";
    }
    byte[] bytes = data.reinterpret(length).toArray(JAVA_BYTE);
    return new String(bytes, StandardCharsets.UTF_8);
  }

  // ---- shader / pipeline registration ------------------------------------

  /**
   * Builds a compute pipeline named {@code name} from {@code wgsl}, with
   * {@code readOnly.length} storage-buffer bindings {@code 0..n-1} (types
   * chosen per {@code readOnly}, all visible to the compute stage). A no-op
   * if {@code name} is already registered.
   */
  static synchronized void registerPipeline(String name, String wgsl, boolean[] readOnly) {
    init();
    if (PIPELINES.containsKey(name)) {
      return;
    }
    try {
      int n = readOnly.length;
      try (Arena a = Arena.ofConfined()) {
        byte[] wgslBytes = wgsl.getBytes(StandardCharsets.UTF_8);
        MemorySegment wgslSrc = a.allocate(wgslBytes.length);
        MemorySegment.copy(wgslBytes, 0, wgslSrc, JAVA_BYTE, 0, wgslBytes.length);

        // WGPUShaderSourceWGSL: {chain:{next8,sType4+pad4}=16, code:StringView16} = 32 bytes
        MemorySegment chain = a.allocate(32);
        chain.set(ADDRESS, 0, MemorySegment.NULL);
        chain.set(JAVA_INT, 8, STYPE_SHADER_SOURCE_WGSL);
        chain.set(ADDRESS, 16, wgslSrc);
        chain.set(JAVA_LONG, 24, wgslBytes.length);

        // WGPUShaderModuleDescriptor: {nextInChain8, label:StringView16} = 24 bytes
        MemorySegment smDesc = a.allocate(24);
        smDesc.set(ADDRESS, 0, chain);
        nullLabel(smDesc, 8);
        MemorySegment shaderModule = (MemorySegment) CREATE_SHADER_MODULE.invoke(device, smDesc);
        if (shaderModule.equals(MemorySegment.NULL)) {
          throw new RuntimeException("wgpuDeviceCreateShaderModule failed for kernel " + name + "\n--- WGSL ---\n" + wgsl);
        }

        // WGPUBindGroupLayoutEntry, 120 bytes each (see webgpu.h for the field-by-field derivation).
        long entrySize = 120;
        MemorySegment entries = a.allocate((long) n * entrySize);
        for (int i = 0; i < n; i++) {
          long base = (long) i * entrySize;
          entries.set(JAVA_INT, base + 8, i);                       // binding
          entries.set(JAVA_LONG, base + 16, SHADER_STAGE_COMPUTE);  // visibility
          entries.set(JAVA_INT, base + 40,                          // buffer.type
              readOnly[i] ? BUFFER_BINDING_TYPE_READ_ONLY_STORAGE : BUFFER_BINDING_TYPE_STORAGE);
          // sampler/texture/storageTexture stay zero => BindingNotUsed
        }
        MemorySegment bglDesc = a.allocate(40); // {nextInChain8, label16, entryCount8, entries8}
        nullLabel(bglDesc, 8);
        bglDesc.set(JAVA_LONG, 24, n);
        bglDesc.set(ADDRESS, 32, entries);
        MemorySegment bgl = (MemorySegment) CREATE_BIND_GROUP_LAYOUT.invoke(device, bglDesc);

        MemorySegment bglHandle = a.allocate(ADDRESS);
        bglHandle.set(ADDRESS, 0, bgl);
        MemorySegment plDesc = a.allocate(48); // {nextInChain8, label16, count8, ptr8, immediateSize4+pad4}
        nullLabel(plDesc, 8);
        plDesc.set(JAVA_LONG, 24, 1L);
        plDesc.set(ADDRESS, 32, bglHandle);
        MemorySegment pl = (MemorySegment) CREATE_PIPELINE_LAYOUT.invoke(device, plDesc);

        byte[] entryBytes = "main".getBytes(StandardCharsets.UTF_8);
        MemorySegment entryPoint = a.allocate(entryBytes.length);
        MemorySegment.copy(entryBytes, 0, entryPoint, JAVA_BYTE, 0, entryBytes.length);

        // WGPUComputePipelineDescriptor: {nextInChain8, label16, layout8, compute:{next8,module8,entryPoint16,constCount8,constants8}=48} = 80
        MemorySegment cpDesc = a.allocate(80);
        nullLabel(cpDesc, 8);
        cpDesc.set(ADDRESS, 24, pl);
        cpDesc.set(ADDRESS, 40, shaderModule);
        cpDesc.set(ADDRESS, 48, entryPoint);
        cpDesc.set(JAVA_LONG, 56, entryBytes.length);
        MemorySegment pipeline = (MemorySegment) CREATE_COMPUTE_PIPELINE.invoke(device, cpDesc);
        if (pipeline.equals(MemorySegment.NULL)) {
          throw new RuntimeException("wgpuDeviceCreateComputePipeline failed for kernel " + name);
        }
        PIPELINES.put(name, new Pipeline(pipeline, bgl));
      }
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  /** Sets a {@code WGPUStringView} field at {@code offset} to the null value ({@code {NULL, SIZE_MAX}}). */
  private static void nullLabel(MemorySegment struct, long offset) {
    struct.set(ADDRESS, offset, MemorySegment.NULL);
    struct.set(JAVA_LONG, offset + 8, WGPU_STRLEN);
  }

  private static MemorySegment buildBindGroup(MemorySegment bindGroupLayout, long[] buffers) throws Throwable {
    int n = buffers.length;
    try (Arena a = Arena.ofConfined()) {
      // WGPUBindGroupEntry, 56 bytes each: {next8, binding4+pad4, buffer8, offset8, size8, sampler8, textureView8}
      MemorySegment entries = a.allocate((long) n * 56);
      for (int i = 0; i < n; i++) {
        long base = (long) i * 56;
        entries.set(JAVA_INT, base + 8, i);
        entries.set(ADDRESS, base + 16, ptr(buffers[i]));
        entries.set(JAVA_LONG, base + 32, WGPU_WHOLE_SIZE);
      }
      MemorySegment desc = a.allocate(48); // {nextInChain8, label16, layout8, entryCount8, entries8}
      nullLabel(desc, 8);
      desc.set(ADDRESS, 24, bindGroupLayout);
      desc.set(JAVA_LONG, 32, n);
      desc.set(ADDRESS, 40, entries);
      MemorySegment bindGroup = (MemorySegment) CREATE_BIND_GROUP.invoke(device, desc);
      if (bindGroup.equals(MemorySegment.NULL)) {
        throw new RuntimeException("wgpuDeviceCreateBindGroup failed");
      }
      return bindGroup;
    }
  }

  // ---- buffers ------------------------------------------------------------

  static long alloc(long bytes) {
    init();
    try {
      return createBuffer(bytes, BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_SRC | BUFFER_USAGE_COPY_DST);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static long createBuffer(long bytes, long usage) throws Throwable {
    try (Arena a = Arena.ofConfined()) {
      MemorySegment desc = a.allocate(48);
      nullLabel(desc, 8);
      desc.set(JAVA_LONG, 24, usage);
      desc.set(JAVA_LONG, 32, bytes);
      MemorySegment buf = (MemorySegment) CREATE_BUFFER.invoke(device, desc);
      if (buf.equals(MemorySegment.NULL)) {
        throw new RuntimeException("wgpuDeviceCreateBuffer failed for " + bytes + " bytes");
      }
      return buf.address();
    }
  }

  static synchronized void free(long buffer) {
    try {
      BUFFER_DESTROY.invoke(ptr(buffer));
      BUFFER_RELEASE.invoke(ptr(buffer));
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static void writeFloats(long buffer, float[] data) {
    init();
    try (Arena a = Arena.ofConfined()) {
      MemorySegment seg = a.allocate((long) data.length * Float.BYTES);
      MemorySegment.copy(data, 0, seg, JAVA_FLOAT, 0, data.length);
      QUEUE_WRITE_BUFFER.invoke(queue, ptr(buffer), 0L, seg, (long) data.length * Float.BYTES);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static synchronized float[] readFloats(long buffer, int count) {
    init();
    long bytes = (long) count * Float.BYTES;
    try {
      ensureReadback(bytes);
      try (Arena a = Arena.ofConfined()) {
        MemorySegment encoder = (MemorySegment) CREATE_COMMAND_ENCODER.invoke(device, MemorySegment.NULL);
        COPY_BUFFER_TO_BUFFER.invoke(encoder, ptr(buffer), 0L, ptr(readbackBuffer), 0L, bytes);
        MemorySegment cmd = (MemorySegment) COMMAND_ENCODER_FINISH.invoke(encoder, MemorySegment.NULL);
        MemorySegment cmdHandle = a.allocate(ADDRESS);
        cmdHandle.set(ADDRESS, 0, cmd);
        QUEUE_SUBMIT.invoke(queue, 1L, cmdHandle);
        DEVICE_POLL.invoke(device, 1, MemorySegment.NULL);
        COMMAND_BUFFER_RELEASE.invoke(cmd);
        COMMAND_ENCODER_RELEASE.invoke(encoder);

        mapDone = false;
        mapStatus = 0;
        MemorySegment cb = a.allocate(CALLBACK_INFO);
        cb.set(ADDRESS, 0, MemorySegment.NULL);
        cb.set(JAVA_INT, 8, CALLBACK_MODE_ALLOW_PROCESS_EVENTS);
        cb.set(ADDRESS, 16, MAP_CALLBACK_STUB);
        cb.set(ADDRESS, 24, MemorySegment.NULL);
        cb.set(ADDRESS, 32, MemorySegment.NULL);
        BUFFER_MAP_ASYNC.invoke(ptr(readbackBuffer), MAP_MODE_READ, 0L, bytes, cb);
        waitFor(() -> mapDone, "buffer map");
        if (mapStatus != MAP_ASYNC_STATUS_SUCCESS) {
          throw new RuntimeException("wgpuBufferMapAsync failed: status " + mapStatus);
        }

        MemorySegment mapped = ((MemorySegment) BUFFER_GET_CONST_MAPPED_RANGE.invoke(ptr(readbackBuffer), 0L, bytes))
            .reinterpret(bytes);
        float[] out = new float[count];
        MemorySegment.copy(mapped, JAVA_FLOAT, 0, out, 0, count);
        BUFFER_UNMAP.invoke(ptr(readbackBuffer));
        return out;
      }
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static void ensureReadback(long bytes) throws Throwable {
    if (readbackBuffer != 0 && readbackCapacity >= bytes) {
      return;
    }
    if (readbackBuffer != 0) {
      BUFFER_DESTROY.invoke(ptr(readbackBuffer));
      BUFFER_RELEASE.invoke(ptr(readbackBuffer));
    }
    readbackBuffer = createBuffer(bytes, BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST);
    readbackCapacity = bytes;
  }

  // ---- dispatch -------------------------------------------------------------

  /**
   * Submits one dispatch of {@code kernel} and blocks until it retires.
   * {@code buffers} are bound {@code 0..n-1}; the last one is the read-only
   * push-constant-equivalent storage buffer, refreshed from {@code pushInts}
   * first.
   */
  static synchronized void dispatch(String kernel, int groupsX, int groupsY, int groupsZ, long[] buffers, int[] pushInts) {
    init();
    Pipeline pipeline = PIPELINES.get(kernel);
    if (pipeline == null) {
      throw new IllegalStateException("unregistered wgpu kernel: " + kernel);
    }
    try {
      if (pushInts != null) {
        long pushBuffer = buffers[buffers.length - 1];
        try (Arena a = Arena.ofConfined()) {
          MemorySegment seg = a.allocate((long) pushInts.length * Integer.BYTES);
          MemorySegment.copy(pushInts, 0, seg, JAVA_INT, 0, pushInts.length);
          QUEUE_WRITE_BUFFER.invoke(queue, ptr(pushBuffer), 0L, seg, (long) pushInts.length * Integer.BYTES);
        }
      }
      MemorySegment bindGroup = pipeline.bindGroupFor(buffers);
      try (Arena a = Arena.ofConfined()) {
        MemorySegment encoder = (MemorySegment) CREATE_COMMAND_ENCODER.invoke(device, MemorySegment.NULL);
        MemorySegment pass = (MemorySegment) BEGIN_COMPUTE_PASS.invoke(encoder, MemorySegment.NULL);
        COMPUTE_SET_PIPELINE.invoke(pass, pipeline.handle);
        COMPUTE_SET_BIND_GROUP.invoke(pass, 0, bindGroup, 0L, MemorySegment.NULL);
        COMPUTE_DISPATCH.invoke(pass, groupsX, groupsY, groupsZ);
        COMPUTE_PASS_END.invoke(pass);
        MemorySegment cmd = (MemorySegment) COMMAND_ENCODER_FINISH.invoke(encoder, MemorySegment.NULL);
        MemorySegment cmdHandle = a.allocate(ADDRESS);
        cmdHandle.set(ADDRESS, 0, cmd);
        QUEUE_SUBMIT.invoke(queue, 1L, cmdHandle);
        DEVICE_POLL.invoke(device, 1, MemorySegment.NULL);

        COMPUTE_PASS_RELEASE.invoke(pass);
        COMMAND_BUFFER_RELEASE.invoke(cmd);
        COMMAND_ENCODER_RELEASE.invoke(encoder);
      }
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static synchronized void sync() {
    if (!ready) {
      return;
    }
    try {
      DEVICE_POLL.invoke(device, 1, MemorySegment.NULL);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }
}
