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
package com.nablatensor.backend.opencl;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Minimal OpenCL host binding via {@link java.lang.foreign}: the Khronos ICD
 * loader ({@code libOpenCL.so.1}) is dlopened, a single device and in-order
 * command queue are created lazily, kernels are built from source with
 * {@code clBuildProgram}, and a launch is one {@code clEnqueueNDRangeKernel}.
 * A direct analogue of the ROCm backend's {@code HipRuntime}: no JNI, no native
 * jars, custom kernels compiled at runtime.
 *
 * <p>All work is issued on one in-order queue, so kernel-after-kernel and
 * copy-after-kernel are implicitly ordered and a blocking
 * {@code clEnqueueReadBuffer} sees every prior result.
 */
final class OpenClRuntime {

  private OpenClRuntime() {
  }

  private static final Arena LIB_ARENA = Arena.ofShared();
  private static final Linker LINKER = Linker.nativeLinker();

  // ---- OpenCL constants we use ----------------------------------------------

  private static final int CL_SUCCESS = 0;
  private static final long CL_DEVICE_TYPE_GPU = 1L << 2;
  private static final long CL_DEVICE_TYPE_ALL = 0xFFFFFFFFL;
  private static final long CL_MEM_READ_WRITE = 1L << 0;
  private static final int CL_TRUE = 1;

  private static final int CL_PLATFORM_NAME = 0x0902;
  private static final int CL_DEVICE_NAME = 0x102B;
  private static final int CL_DEVICE_VERSION = 0x102F;
  private static final int CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;
  private static final int CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032;
  private static final int CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
  private static final int CL_PROGRAM_BUILD_LOG = 0x1183;

  // ---- symbol lookup ------------------------------------------------------

  private static final SymbolLookup CL = clLookup();

  private static SymbolLookup clLookup() {
    String explicit = System.getenv("OPENCL_LIBRARY_PATH");
    if (explicit != null && !explicit.isBlank()) {
      return SymbolLookup.libraryLookup(explicit, LIB_ARENA);
    }
    for (String candidate : new String[] {"libOpenCL.so.1", "libOpenCL.so", "OpenCL"}) {
      try {
        return SymbolLookup.libraryLookup(candidate, LIB_ARENA);
      } catch (RuntimeException ignored) {
        // try the next candidate
      }
    }
    return SymbolLookup.libraryLookup("libOpenCL.so.1", LIB_ARENA); // surface the real error
  }

  private static MethodHandle cl(String name, FunctionDescriptor descriptor) {
    return LINKER.downcallHandle(CL.find(name).orElseThrow(() -> new UnsatisfiedLinkError(name)), descriptor);
  }

  private static MethodHandle clOptional(String name, FunctionDescriptor descriptor) {
    return CL.find(name).map(symbol -> LINKER.downcallHandle(symbol, descriptor)).orElse(null);
  }

  // cl_int clGetPlatformIDs(cl_uint, cl_platform_id*, cl_uint*)
  private static final MethodHandle GET_PLATFORM_IDS =
      cl("clGetPlatformIDs", FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle GET_PLATFORM_INFO =
      cl("clGetPlatformInfo", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));
  // cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*)
  private static final MethodHandle GET_DEVICE_IDS =
      cl("clGetDeviceIDs", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle GET_DEVICE_INFO =
      cl("clGetDeviceInfo", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));
  // cl_context clCreateContext(const cl_context_properties*, cl_uint, const cl_device_id*, fn, void*, cl_int*)
  private static final MethodHandle CREATE_CONTEXT =
      cl("clCreateContext", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
  // cl_command_queue clCreateCommandQueueWithProperties(cl_context, cl_device_id, const cl_queue_properties*, cl_int*)
  private static final MethodHandle CREATE_QUEUE_WITH_PROPERTIES =
      clOptional("clCreateCommandQueueWithProperties",
          FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
  // cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties, cl_int*)
  private static final MethodHandle CREATE_QUEUE =
      clOptional("clCreateCommandQueue",
          FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
  // cl_program clCreateProgramWithSource(cl_context, cl_uint, const char**, const size_t*, cl_int*)
  private static final MethodHandle CREATE_PROGRAM =
      cl("clCreateProgramWithSource", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
  // cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id*, const char*, fn, void*)
  private static final MethodHandle BUILD_PROGRAM =
      cl("clBuildProgram", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle GET_PROGRAM_BUILD_INFO =
      cl("clGetProgramBuildInfo", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_KERNEL =
      cl("clCreateKernel", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS));
  // cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void*)
  private static final MethodHandle SET_KERNEL_ARG =
      cl("clSetKernelArg", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS));
  // cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void*, cl_int*)
  private static final MethodHandle CREATE_BUFFER =
      cl("clCreateBuffer", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS, ADDRESS));
  // cl_int clEnqueueWriteBuffer(queue, mem, blocking, offset, size, ptr, nEvents, events, event)
  private static final MethodHandle ENQUEUE_WRITE =
      cl("clEnqueueWriteBuffer", FunctionDescriptor.of(JAVA_INT,
          ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle ENQUEUE_READ =
      cl("clEnqueueReadBuffer", FunctionDescriptor.of(JAVA_INT,
          ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
  // cl_int clEnqueueNDRangeKernel(queue, kernel, workDim, globalOffset, globalSize, localSize, nEvents, events, event)
  private static final MethodHandle ENQUEUE_NDRANGE =
      cl("clEnqueueNDRangeKernel", FunctionDescriptor.of(JAVA_INT,
          ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle FINISH =
      cl("clFinish", FunctionDescriptor.of(JAVA_INT, ADDRESS));
  private static final MethodHandle RELEASE_MEM =
      cl("clReleaseMemObject", FunctionDescriptor.of(JAVA_INT, ADDRESS));
  private static final MethodHandle RELEASE_KERNEL =
      cl("clReleaseKernel", FunctionDescriptor.of(JAVA_INT, ADDRESS));

  // ---- error handling -----------------------------------------------------

  private static void check(int status, String operation) {
    if (status != CL_SUCCESS) {
      throw new RuntimeException(operation + " failed: OpenCL error " + status);
    }
  }

  private static RuntimeException wrap(Throwable failure) {
    return failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
  }

  // ---- probe / context --------------------------------------------------

  static boolean probe() {
    try {
      return firstGpuDevice() != null;
    } catch (Throwable failure) {
      return false;
    }
  }

  record DeviceInfo(MemorySegment platform, MemorySegment device, String platformName,
                    String name, String version, int maxWorkGroupSize, boolean fp64,
                    long globalMemBytes) {
  }

  private static DeviceInfo sharedContext;
  private static MemorySegment context;
  private static MemorySegment queue;

  static synchronized DeviceInfo context() {
    if (sharedContext == null) {
      init();
    }
    return sharedContext;
  }

  static synchronized MemorySegment queue() {
    if (queue == null) {
      init();
    }
    return queue;
  }

  private static void init() {
    try {
      DeviceInfo info = firstGpuDevice();
      if (info == null) {
        throw new IllegalStateException("no OpenCL GPU device found");
      }
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment devices = arena.allocate(ADDRESS);
        devices.set(ADDRESS, 0, info.device());
        MemorySegment err = arena.allocate(JAVA_INT);
        MemorySegment ctx = (MemorySegment) CREATE_CONTEXT.invoke(
            MemorySegment.NULL, 1, devices, MemorySegment.NULL, MemorySegment.NULL, err);
        check(err.get(JAVA_INT, 0), "clCreateContext");
        MemorySegment q;
        if (CREATE_QUEUE_WITH_PROPERTIES != null) {
          q = (MemorySegment) CREATE_QUEUE_WITH_PROPERTIES.invoke(
              ctx, info.device(), MemorySegment.NULL, err);
        } else {
          q = (MemorySegment) CREATE_QUEUE.invoke(ctx, info.device(), 0L, err);
        }
        check(err.get(JAVA_INT, 0), "clCreateCommandQueue");
        context = ctx.reinterpret(Long.MAX_VALUE);
        queue = q.reinterpret(Long.MAX_VALUE);
        sharedContext = info;
      }
    } catch (Throwable failure) {
      throw wrap(failure);
    }
  }

  /**
   * The first GPU device on the first platform that has one, or — if a platform
   * or index is pinned with {@code nablatensor.opencl.platform} /
   * {@code nablatensor.opencl.device} — that exact choice. Falls back to any
   * device type when no platform exposes a GPU (so a CPU ICD such as PoCL is
   * still usable for correctness runs).
   */
  private static DeviceInfo firstGpuDevice() throws Throwable {
    String platformFilter = System.getProperty("nablatensor.opencl.platform",
        System.getenv().getOrDefault("NABLATENSOR_OPENCL_PLATFORM", "")).toLowerCase();
    int deviceIndex = Integer.getInteger("nablatensor.opencl.device", 0);

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment countOut = arena.allocate(JAVA_INT);
      if ((int) GET_PLATFORM_IDS.invoke(0, MemorySegment.NULL, countOut) != CL_SUCCESS) {
        return null;
      }
      int platformCount = countOut.get(JAVA_INT, 0);
      if (platformCount <= 0) {
        return null;
      }
      MemorySegment platforms = arena.allocate(ADDRESS.byteSize() * platformCount);
      check((int) GET_PLATFORM_IDS.invoke(platformCount, platforms, MemorySegment.NULL), "clGetPlatformIDs");

      DeviceInfo anyDevice = null;
      for (long deviceType : new long[] {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL}) {
        for (int p = 0; p < platformCount; p++) {
          MemorySegment platform = platforms.getAtIndex(ADDRESS, p);
          String platformName = infoString(GET_PLATFORM_INFO, platform, CL_PLATFORM_NAME, arena);
          if (!platformFilter.isBlank() && !platformName.toLowerCase().contains(platformFilter)) {
            continue;
          }
          if ((int) GET_DEVICE_IDS.invoke(platform, deviceType, 0, MemorySegment.NULL, countOut) != CL_SUCCESS) {
            continue;
          }
          int deviceCount = countOut.get(JAVA_INT, 0);
          if (deviceCount <= 0) {
            continue;
          }
          MemorySegment devices = arena.allocate(ADDRESS.byteSize() * deviceCount);
          check((int) GET_DEVICE_IDS.invoke(platform, deviceType, deviceCount, devices, MemorySegment.NULL),
              "clGetDeviceIDs");
          int idx = Math.min(Math.max(deviceIndex, 0), deviceCount - 1);
          MemorySegment device = devices.getAtIndex(ADDRESS, idx);
          DeviceInfo info = describe(platform, device, platformName, arena);
          if (deviceType == CL_DEVICE_TYPE_GPU) {
            return info;
          }
          if (anyDevice == null) {
            anyDevice = info;
          }
        }
        if (anyDevice != null) {
          return anyDevice;
        }
      }
      return null;
    }
  }

  private static DeviceInfo describe(MemorySegment platform, MemorySegment device,
                                     String platformName, Arena arena) throws Throwable {
    String name = infoString(GET_DEVICE_INFO, device, CL_DEVICE_NAME, arena);
    String version = infoString(GET_DEVICE_INFO, device, CL_DEVICE_VERSION, arena);
    MemorySegment scratch = arena.allocate(JAVA_LONG);
    check((int) GET_DEVICE_INFO.invoke(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, JAVA_LONG.byteSize(),
        scratch, MemorySegment.NULL), "clGetDeviceInfo(MAX_WORK_GROUP_SIZE)");
    int maxWorkGroup = (int) Math.min(Integer.MAX_VALUE, scratch.get(JAVA_LONG, 0));
    long fp64Config = 0;
    if ((int) GET_DEVICE_INFO.invoke(device, CL_DEVICE_DOUBLE_FP_CONFIG, JAVA_LONG.byteSize(),
        scratch, MemorySegment.NULL) == CL_SUCCESS) {
      fp64Config = scratch.get(JAVA_LONG, 0);
    }
    long globalMem = 0;
    if ((int) GET_DEVICE_INFO.invoke(device, CL_DEVICE_GLOBAL_MEM_SIZE, JAVA_LONG.byteSize(),
        scratch, MemorySegment.NULL) == CL_SUCCESS) {
      globalMem = scratch.get(JAVA_LONG, 0);
    }
    return new DeviceInfo(platform, device, platformName, name, version,
        maxWorkGroup, fp64Config != 0, globalMem);
  }

  private static String infoString(MethodHandle infoCall, MemorySegment object, int param, Arena arena)
      throws Throwable {
    MemorySegment sizeOut = arena.allocate(JAVA_LONG);
    if ((int) infoCall.invoke(object, param, 0L, MemorySegment.NULL, sizeOut) != CL_SUCCESS) {
      return "";
    }
    long size = sizeOut.get(JAVA_LONG, 0);
    if (size <= 0) {
      return "";
    }
    MemorySegment buffer = arena.allocate(size);
    if ((int) infoCall.invoke(object, param, size, buffer, MemorySegment.NULL) != CL_SUCCESS) {
      return "";
    }
    return buffer.getString(0);
  }

  static void synchronize() {
    try {
      check((int) FINISH.invoke(queue()), "clFinish");
    } catch (Throwable failure) {
      throw wrap(failure);
    }
  }

  // ---- kernels ----------------------------------------------------------

  static long compile(String source, String kernelName, String buildOptions) {
    context();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment src = arena.allocateFrom(source);
      MemorySegment strings = arena.allocate(ADDRESS);
      strings.set(ADDRESS, 0, src);
      MemorySegment err = arena.allocate(JAVA_INT);
      MemorySegment program = (MemorySegment) CREATE_PROGRAM.invoke(
          context, 1, strings, MemorySegment.NULL, err);
      check(err.get(JAVA_INT, 0), "clCreateProgramWithSource");

      MemorySegment devices = arena.allocate(ADDRESS);
      devices.set(ADDRESS, 0, sharedContext.device());
      MemorySegment options = buildOptions == null || buildOptions.isBlank()
          ? MemorySegment.NULL : arena.allocateFrom(buildOptions);
      int status = (int) BUILD_PROGRAM.invoke(program, 1, devices, options,
          MemorySegment.NULL, MemorySegment.NULL);
      if (status != CL_SUCCESS) {
        throw new RuntimeException("clBuildProgram failed (error " + status + "):\n"
            + buildLog(program, arena));
      }
      MemorySegment kernErr = arena.allocate(JAVA_INT);
      MemorySegment kernel = (MemorySegment) CREATE_KERNEL.invoke(
          program, arena.allocateFrom(kernelName), kernErr);
      check(kernErr.get(JAVA_INT, 0), "clCreateKernel");
      // The program is left retained for the life of the process: kernels are
      // cached by source, so the count of live programs is bounded and small.
      return kernel.reinterpret(Long.MAX_VALUE).address();
    } catch (Throwable failure) {
      throw wrap(failure);
    }
  }

  private static String buildLog(MemorySegment program, Arena arena) throws Throwable {
    MemorySegment sizeOut = arena.allocate(JAVA_LONG);
    if ((int) GET_PROGRAM_BUILD_INFO.invoke(program, sharedContext.device(), CL_PROGRAM_BUILD_LOG,
        0L, MemorySegment.NULL, sizeOut) != CL_SUCCESS) {
      return "";
    }
    long size = sizeOut.get(JAVA_LONG, 0);
    if (size <= 0) {
      return "";
    }
    MemorySegment buffer = arena.allocate(size + 1);
    GET_PROGRAM_BUILD_INFO.invoke(program, sharedContext.device(), CL_PROGRAM_BUILD_LOG,
        size, buffer, MemorySegment.NULL);
    return buffer.getString(0);
  }

  static void releaseKernel(long kernel) {
    try {
      RELEASE_KERNEL.invoke(MemorySegment.ofAddress(kernel));
    } catch (Throwable ignored) {
      // teardown best-effort
    }
  }

  // ---- memory ---------------------------------------------------------

  static long malloc(long bytes) {
    context();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment err = arena.allocate(JAVA_INT);
      MemorySegment mem = (MemorySegment) CREATE_BUFFER.invoke(
          context, CL_MEM_READ_WRITE, Math.max(1, bytes), MemorySegment.NULL, err);
      check(err.get(JAVA_INT, 0), "clCreateBuffer");
      return mem.reinterpret(Long.MAX_VALUE).address();
    } catch (Throwable failure) {
      throw wrap(failure);
    }
  }

  static void free(long mem) {
    try {
      check((int) RELEASE_MEM.invoke(MemorySegment.ofAddress(mem)), "clReleaseMemObject");
    } catch (Throwable failure) {
      throw wrap(failure);
    }
  }

  static void uploadDoubles(long mem, double[] data) {
    long bytes = (long) data.length * Double.BYTES;
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment host = arena.allocateFrom(JAVA_DOUBLE, data);
      check((int) ENQUEUE_WRITE.invoke(queue(), MemorySegment.ofAddress(mem), CL_TRUE, 0L, bytes,
          host, 0, MemorySegment.NULL, MemorySegment.NULL), "clEnqueueWriteBuffer");
    } catch (Throwable failure) {
      throw wrap(failure);
    }
  }

  static double[] downloadDoubles(long mem, int count) {
    long bytes = (long) count * Double.BYTES;
    double[] out = new double[count];
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment host = arena.allocate(bytes, JAVA_DOUBLE.byteSize());
      check((int) ENQUEUE_READ.invoke(queue(), MemorySegment.ofAddress(mem), CL_TRUE, 0L, bytes,
          host, 0, MemorySegment.NULL, MemorySegment.NULL), "clEnqueueReadBuffer");
      MemorySegment.copy(host, JAVA_DOUBLE, 0, out, 0, count);
    } catch (Throwable failure) {
      throw wrap(failure);
    }
    return out;
  }

  // ---- launch -------------------------------------------------------

  private static final int MAX_KERNEL_ARGUMENTS = 16;

  /**
   * Launch a 1-D range of {@code groups} work-groups of {@code local} work-items.
   * Arguments are matched positionally: {@link Long} is a device buffer handle or
   * a 64-bit ({@code ulong}) scalar, {@link Integer} a 32-bit scalar,
   * {@link Float} a 32-bit float.
   */
  static synchronized void launch(long kernel, int groups, int local, Object... arguments) {
    if (arguments.length > MAX_KERNEL_ARGUMENTS) {
      throw new IllegalArgumentException("too many kernel arguments: " + arguments.length);
    }
    MemorySegment kern = MemorySegment.ofAddress(kernel);
    try (Arena arena = Arena.ofConfined()) {
      for (int i = 0; i < arguments.length; i++) {
        MemorySegment slot;
        long size;
        switch (arguments[i]) {
          case Long value -> {
            slot = arena.allocate(JAVA_LONG);
            slot.set(JAVA_LONG, 0, value);
            size = JAVA_LONG.byteSize();
          }
          case Integer value -> {
            slot = arena.allocate(JAVA_INT);
            slot.set(JAVA_INT, 0, value);
            size = JAVA_INT.byteSize();
          }
          case Float value -> {
            slot = arena.allocate(JAVA_FLOAT);
            slot.set(JAVA_FLOAT, 0, value);
            size = JAVA_FLOAT.byteSize();
          }
          default -> throw new IllegalArgumentException("unsupported kernel argument: " + arguments[i]);
        }
        check((int) SET_KERNEL_ARG.invoke(kern, i, size, slot), "clSetKernelArg(" + i + ")");
      }
      MemorySegment global = arena.allocate(JAVA_LONG);
      global.set(JAVA_LONG, 0, (long) groups * local);
      MemorySegment localSize = arena.allocate(JAVA_LONG);
      localSize.set(JAVA_LONG, 0, local);
      check((int) ENQUEUE_NDRANGE.invoke(queue(), kern, 1, MemorySegment.NULL, global, localSize,
          0, MemorySegment.NULL, MemorySegment.NULL), "clEnqueueNDRangeKernel");
    } catch (Throwable failure) {
      throw wrap(failure);
    }
  }
}
