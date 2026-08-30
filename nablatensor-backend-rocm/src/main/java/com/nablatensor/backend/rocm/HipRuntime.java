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
package com.nablatensor.backend.rocm;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Minimal AMD ROCm/HIP runtime + HIPRTC binding via {@link java.lang.foreign}.
 * A direct analogue of the CUDA backend's {@code CudaRuntime}: no JNI, no native
 * jars, custom kernels compiled at runtime.
 *
 * <p>Work is issued on the null stream, so kernel-after-kernel and
 * copy-after-kernel are implicitly ordered and a {@code hipMemcpyDtoH} sees all
 * prior results. There is no stream/graph machinery here yet (Phase 7).
 */
final class HipRuntime {

  private HipRuntime() {
  }

  private static final Arena LIB_ARENA = Arena.ofShared();
  private static final Linker LINKER = Linker.nativeLinker();

  // ---- HIP runtime (libamdhip64) ---------------------------------------

  private static final SymbolLookup HIP = hipLookup();

  private static SymbolLookup hipLookup() {
    String explicit = System.getenv("HIP_RUNTIME_PATH");
    if (explicit != null && !explicit.isBlank()) {
      return SymbolLookup.libraryLookup(explicit, LIB_ARENA);
    }
    for (String candidate : new String[] {"/opt/rocm/lib/libamdhip64.so", "libamdhip64.so"}) {
      try {
        return SymbolLookup.libraryLookup(candidate, LIB_ARENA);
      } catch (RuntimeException ignored) {
        // try the next candidate
      }
    }
    return SymbolLookup.libraryLookup("libamdhip64.so", LIB_ARENA); // surface the real error
  }

  private static MethodHandle hip(String name, FunctionDescriptor descriptor) {
    return LINKER.downcallHandle(HIP.find(name).orElseThrow(() -> new UnsatisfiedLinkError(name)), descriptor);
  }

  private static MethodHandle hipOptional(String name, FunctionDescriptor descriptor) {
    return HIP.find(name).map(symbol -> LINKER.downcallHandle(symbol, descriptor)).orElse(null);
  }

  private static final MethodHandle INIT = hip("hipInit", FunctionDescriptor.of(JAVA_INT, JAVA_INT));
  private static final MethodHandle DEVICE_COUNT = hip("hipGetDeviceCount", FunctionDescriptor.of(JAVA_INT, ADDRESS));
  private static final MethodHandle SET_DEVICE = hip("hipSetDevice", FunctionDescriptor.of(JAVA_INT, JAVA_INT));
  private static final MethodHandle DEVICE_SYNC = hip("hipDeviceSynchronize", FunctionDescriptor.of(JAVA_INT));
  private static final MethodHandle DEVICE_GET = hip("hipDeviceGet", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
  private static final MethodHandle DEVICE_NAME = hip("hipDeviceGetName", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT));
  private static final MethodHandle MALLOC = hip("hipMalloc", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
  private static final MethodHandle FREE = hip("hipFree", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
  private static final MethodHandle MEMCPY_H2D = hip("hipMemcpyHtoD", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG));
  private static final MethodHandle MEMCPY_D2H = hip("hipMemcpyDtoH", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG));
  private static final MethodHandle MEMCPY_D2D = hip("hipMemcpyDtoD", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG));
  private static final MethodHandle MODULE_LOAD = hip("hipModuleLoadData", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle MODULE_FUNCTION = hip("hipModuleGetFunction", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
  private static final MethodHandle LAUNCH = hip("hipModuleLaunchKernel", FunctionDescriptor.of(JAVA_INT,
      JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));
  private static final MethodHandle ERROR_STRING = hip("hipGetErrorString", FunctionDescriptor.of(ADDRESS, JAVA_INT));

  // ---- HIPRTC (libhiprtc) --------------------------------------------------

  private static final SymbolLookup HIPRTC = hiprtcLookup();

  private static SymbolLookup hiprtcLookup() {
    String explicit = System.getenv("HIPRTC_PATH");
    if (explicit != null && !explicit.isBlank()) {
      return SymbolLookup.libraryLookup(explicit, LIB_ARENA);
    }
    for (String candidate : new String[] {"/opt/rocm/lib/libhiprtc.so", "libhiprtc.so"}) {
      try {
        return SymbolLookup.libraryLookup(candidate, LIB_ARENA);
      } catch (RuntimeException ignored) {
        // try the next candidate
      }
    }
    return SymbolLookup.libraryLookup("libhiprtc.so", LIB_ARENA);
  }

  private static MethodHandle hiprtc(String name, FunctionDescriptor descriptor) {
    return LINKER.downcallHandle(HIPRTC.find(name).orElseThrow(() -> new UnsatisfiedLinkError(name)), descriptor);
  }

  private static final MethodHandle RTC_CREATE = hiprtc("hiprtcCreateProgram",
      FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle RTC_COMPILE = hiprtc("hiprtcCompileProgram", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
  private static final MethodHandle RTC_CODE_SIZE = hiprtc("hiprtcGetCodeSize", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle RTC_CODE = hiprtc("hiprtcGetCode", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle RTC_LOG_SIZE = hiprtc("hiprtcGetProgramLogSize", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle RTC_LOG = hiprtc("hiprtcGetProgramLog", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

  // ---- error handling ---------------------------------------------------

  private static void check(int status, String operation) {
    if (status == 0) {
      return;
    }
    String message = "?";
    try {
      MemorySegment text = (MemorySegment) ERROR_STRING.invoke(status);
      message = text.reinterpret(256).getString(0);
    } catch (Throwable ignored) {
      // fall through with the numeric code only
    }
    throw new RuntimeException(operation + " failed: " + message + " (code " + status + ")");
  }

  // ---- lifecycle ------------------------------------------------------------

  static boolean probe() {
    try (Arena arena = Arena.ofConfined()) {
      if ((int) INIT.invoke(0) != 0) {
        return false;
      }
      MemorySegment count = arena.allocate(JAVA_INT);
      if ((int) DEVICE_COUNT.invoke(count) != 0) {
        return false;
      }
      return count.get(JAVA_INT, 0) > 0;
    } catch (Throwable failure) {
      return false;
    }
  }

  record DeviceInfo(int device, String name, String arch) {
  }

  private static DeviceInfo sharedContext;

  static synchronized DeviceInfo context() {
    if (sharedContext == null) {
      sharedContext = initContext(0);
    }
    return sharedContext;
  }

  private static DeviceInfo initContext(int ordinal) {
    try (Arena arena = Arena.ofConfined()) {
      check((int) INIT.invoke(0), "hipInit");
      check((int) SET_DEVICE.invoke(ordinal), "hipSetDevice");
      String name = "";
      try {
        MemorySegment deviceOut = arena.allocate(JAVA_INT);
        MemorySegment nameOut = arena.allocate(256);
        if ((int) DEVICE_GET.invoke(deviceOut, ordinal) == 0
            && (int) DEVICE_NAME.invoke(nameOut, 256, deviceOut.get(JAVA_INT, 0)) == 0) {
          name = nameOut.getString(0);
        }
      } catch (Throwable ignored) {
        // device name is diagnostic only
      }
      return new DeviceInfo(ordinal, name, resolveArch(ordinal, arena));
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static final Pattern GFX = Pattern.compile("gfx[0-9a-fA-F]{3,}");

  /**
   * Determines the {@code --offload-arch} target for HIPRTC. An explicit
   * {@code -Dnablatensor.rocm.arch} / {@code NABLATENSOR_ROCM_ARCH} wins; otherwise the
   * {@code gcnArchName} string is recovered by scanning the raw
   * {@code hipDeviceProp_t} bytes (the struct layout is version-dependent, the
   * {@code "gfxNNNN"} token is not), falling back to {@code gfx1100}.
   */
  private static String resolveArch(int ordinal, Arena arena) {
    String override = System.getProperty("nablatensor.tensor.rocm.arch",
        System.getenv().getOrDefault("NABLATENSOR_ROCM_ARCH", ""));
    if (override != null && !override.isBlank()) {
      return override;
    }
    for (String symbol : new String[] {"hipGetDevicePropertiesR0600", "hipGetDeviceProperties"}) {
      MethodHandle properties = hipOptional(symbol, FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
      if (properties == null) {
        continue;
      }
      try {
        MemorySegment prop = arena.allocate(8192); // hipDeviceProp_t, zero-filled
        if ((int) properties.invoke(prop, ordinal) != 0) {
          continue;
        }
        String ascii = new String(prop.toArray(JAVA_BYTE), StandardCharsets.ISO_8859_1);
        Matcher matcher = GFX.matcher(ascii);
        if (matcher.find()) {
          return matcher.group();
        }
      } catch (Throwable ignored) {
        // try the next symbol / fall back
      }
    }
    return "gfx1100";
  }

  static void synchronize() {
    try {
      check((int) DEVICE_SYNC.invoke(), "hipDeviceSynchronize");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  // ---- memory ------------------------------------------------------------

  private record PendingFree(long pointer) {
  }

  private static final java.util.concurrent.ConcurrentLinkedQueue<PendingFree> PENDING_FREES =
      new java.util.concurrent.ConcurrentLinkedQueue<>();

  static void enqueueFree(long pointer) {
    PENDING_FREES.add(new PendingFree(pointer));
  }

  static void drainPendingFrees() {
    PendingFree pending;
    while ((pending = PENDING_FREES.poll()) != null) {
      try {
        free(pending.pointer());
      } catch (RuntimeException ignored) {
        // a Cleaner-driven free after context teardown is not fatal
      }
    }
  }

  static long malloc(long bytes) {
    drainPendingFrees();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_LONG);
      check((int) MALLOC.invoke(out, bytes), "hipMalloc");
      return out.get(JAVA_LONG, 0);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static void free(long pointer) {
    try {
      check((int) FREE.invoke(pointer), "hipFree");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static long uploadFloats(float[] data) {
    long bytes = (long) data.length * Float.BYTES;
    long pointer = malloc(bytes);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment host = arena.allocateFrom(JAVA_FLOAT, data);
      check((int) MEMCPY_H2D.invoke(pointer, host, bytes), "hipMemcpyHtoD");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
    return pointer;
  }

  static float[] downloadFloats(long pointer, int count) {
    long bytes = (long) count * Float.BYTES;
    float[] out = new float[count];
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment host = arena.allocate(bytes, JAVA_FLOAT.byteSize());
      check((int) MEMCPY_D2H.invoke(host, pointer, bytes), "hipMemcpyDtoH");
      MemorySegment.copy(host, JAVA_FLOAT, 0, out, 0, count);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
    return out;
  }

  /** Host-to-device copy of {@code data} into an already-allocated {@code pointer}. */
  static void uploadDoubles(long pointer, double[] data) {
    long bytes = (long) data.length * Double.BYTES;
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment host = arena.allocateFrom(JAVA_DOUBLE, data);
      check((int) MEMCPY_H2D.invoke(pointer, host, bytes), "hipMemcpyHtoD");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static double[] downloadDoubles(long pointer, int count) {
    long bytes = (long) count * Double.BYTES;
    double[] out = new double[count];
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment host = arena.allocate(bytes, JAVA_DOUBLE.byteSize());
      check((int) MEMCPY_D2H.invoke(host, pointer, bytes), "hipMemcpyDtoH");
      MemorySegment.copy(host, JAVA_DOUBLE, 0, out, 0, count);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
    return out;
  }

  static void copyDevice(long target, long source, long bytes) {
    try {
      check((int) MEMCPY_D2D.invoke(target, source, bytes), "hipMemcpyDtoD");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  // ---- kernels ---------------------------------------------------------------

  static byte[] compile(String source, String arch) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment programOut = arena.allocate(ADDRESS);
      check((int) RTC_CREATE.invoke(programOut, arena.allocateFrom(source),
          arena.allocateFrom("nablatensor_kernels.hip"), 0, MemorySegment.NULL, MemorySegment.NULL),
          "hiprtcCreateProgram");
      MemorySegment program = programOut.get(ADDRESS, 0);
      String[] options = {"--offload-arch=" + arch};
      MemorySegment optionPointers = arena.allocate(ADDRESS.byteSize() * options.length);
      for (int i = 0; i < options.length; i++) {
        optionPointers.setAtIndex(ADDRESS, i, arena.allocateFrom(options[i]));
      }
      int status = (int) RTC_COMPILE.invoke(program, options.length, optionPointers);
      if (status != 0) {
        throw new RuntimeException("hiprtcCompileProgram failed (arch " + arch + "):\n" + programLog(program, arena));
      }
      MemorySegment size = arena.allocate(JAVA_LONG);
      check((int) RTC_CODE_SIZE.invoke(program, size), "hiprtcGetCodeSize");
      long length = size.get(JAVA_LONG, 0);
      MemorySegment buffer = arena.allocate(length);
      check((int) RTC_CODE.invoke(program, buffer), "hiprtcGetCode");
      byte[] code = new byte[Math.toIntExact(length)];
      MemorySegment.copy(buffer, JAVA_BYTE, 0, code, 0, code.length);
      return code;
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static String programLog(MemorySegment program, Arena arena) throws Throwable {
    MemorySegment size = arena.allocate(JAVA_LONG);
    if ((int) RTC_LOG_SIZE.invoke(program, size) != 0) {
      return "";
    }
    long length = size.get(JAVA_LONG, 0);
    if (length <= 0) {
      return "";
    }
    MemorySegment buffer = arena.allocate(length + 1);
    RTC_LOG.invoke(program, buffer);
    return buffer.getString(0);
  }

  static long loadFunction(byte[] code, String kernelName) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment image = arena.allocate(code.length + 1L);
      MemorySegment.copy(code, 0, image, JAVA_BYTE, 0, code.length);
      MemorySegment module = arena.allocate(JAVA_LONG);
      check((int) MODULE_LOAD.invoke(module, image), "hipModuleLoadData");
      MemorySegment function = arena.allocate(JAVA_LONG);
      check((int) MODULE_FUNCTION.invoke(function, module.get(JAVA_LONG, 0), arena.allocateFrom(kernelName)),
          "hipModuleGetFunction");
      return function.get(JAVA_LONG, 0);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static final int MAX_KERNEL_ARGUMENTS = 16;
  private static final Arena LAUNCH_ARENA = Arena.ofAuto();
  private static final MemorySegment LAUNCH_VALUES =
      LAUNCH_ARENA.allocate((long) MAX_KERNEL_ARGUMENTS * JAVA_LONG.byteSize(), JAVA_LONG.byteSize());
  private static final MemorySegment LAUNCH_PARAMETERS =
      LAUNCH_ARENA.allocate((long) MAX_KERNEL_ARGUMENTS * ADDRESS.byteSize(), ADDRESS.byteSize());

  static {
    for (int i = 0; i < MAX_KERNEL_ARGUMENTS; i++) {
      LAUNCH_PARAMETERS.setAtIndex(ADDRESS, i,
          LAUNCH_VALUES.asSlice((long) i * JAVA_LONG.byteSize(), JAVA_LONG.byteSize()));
    }
  }

  /** Launch a 1-D grid. Arguments are Long (device pointers), Integer, or Float. */
  static synchronized void launch(long function, int grid, int block, Object... arguments) {
    launchBlocks(function, grid, 1, block, 1, 1, 0, arguments);
  }

  /** Launch a 1-D grid whose blocks use two thread dimensions. */
  static synchronized void launchBlocks2d(long function, int grid, int blockX, int blockY, Object... arguments) {
    launchBlocks(function, grid, 1, blockX, blockY, 1, 0, arguments);
  }

  private static void launchBlocks(
      long function, int gridX, int gridY, int blockX, int blockY, int blockZ, int sharedBytes, Object[] arguments) {
    prepareLaunchArguments(arguments);
    try {
      check((int) LAUNCH.invoke(function, gridX, gridY, 1, blockX, blockY, blockZ, sharedBytes, 0L,
          LAUNCH_PARAMETERS, MemorySegment.NULL), "hipModuleLaunchKernel");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static void prepareLaunchArguments(Object[] arguments) {
    if (arguments.length > MAX_KERNEL_ARGUMENTS) {
      throw new IllegalArgumentException("too many kernel arguments: " + arguments.length);
    }
    for (int i = 0; i < arguments.length; i++) {
      long offset = (long) i * JAVA_LONG.byteSize();
      switch (arguments[i]) {
        case Long value -> LAUNCH_VALUES.set(JAVA_LONG, offset, value);
        case Integer value -> LAUNCH_VALUES.set(JAVA_INT, offset, value);
        case Float value -> LAUNCH_VALUES.set(JAVA_FLOAT, offset, value);
        default -> throw new IllegalArgumentException("unsupported kernel argument: " + arguments[i]);
      }
    }
  }
}
