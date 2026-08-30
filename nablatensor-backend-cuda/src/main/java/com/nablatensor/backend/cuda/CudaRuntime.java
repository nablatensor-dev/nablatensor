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
package com.nablatensor.backend.cuda;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Minimal CUDA Driver API + NVRTC binding via {@link java.lang.foreign}.
 * A direct java.lang.foreign binding. No JNI, no native jars.
 */
final class CudaRuntime {

  private CudaRuntime() {
  }

  private static final Arena LIB_ARENA = Arena.ofShared();
  private static final Linker LINKER = Linker.nativeLinker();

  // ---- driver (libcuda) ---------------------------------------------------

  private static final SymbolLookup CUDA = driverLookup();
  private static final MethodHandle INIT = driver("cuInit", FunctionDescriptor.of(JAVA_INT, JAVA_INT));
  private static final MethodHandle DEVICE_COUNT = driver("cuDeviceGetCount", FunctionDescriptor.of(JAVA_INT, ADDRESS));
  private static final MethodHandle DEVICE_GET = driver("cuDeviceGet", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
  private static final MethodHandle DEVICE_NAME = driver("cuDeviceGetName", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT));
  private static final MethodHandle DEVICE_ATTRIBUTE = driver("cuDeviceGetAttribute", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT));
  private static final MethodHandle CTX_CREATE = driver("cuCtxCreate", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT));
  private static final MethodHandle CTX_SYNC = driver("cuCtxSynchronize", FunctionDescriptor.of(JAVA_INT));
  private static final MethodHandle MEM_ALLOC = driver("cuMemAlloc", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
  private static final MethodHandle MEM_FREE = driver("cuMemFree", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
  private static final MethodHandle MEMCPY_H2D = driver("cuMemcpyHtoD", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG));
  private static final MethodHandle MEMCPY_D2H = driver("cuMemcpyDtoH", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG));
  private static final MethodHandle MODULE_LOAD = driver("cuModuleLoadData", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle MODULE_FUNCTION = driver("cuModuleGetFunction", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
  private static final MethodHandle LAUNCH = driver("cuLaunchKernel", FunctionDescriptor.of(JAVA_INT, JAVA_LONG,
      JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));
  private static final MethodHandle ERROR_NAME = driver("cuGetErrorName", FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS));
  private static final MethodHandle MEM_GET_INFO = driver("cuMemGetInfo", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle STREAM_CREATE = driver("cuStreamCreate", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
  private static final MethodHandle STREAM_SYNC = driver("cuStreamSynchronize", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
  private static final MethodHandle STREAM_BEGIN_CAPTURE = driver("cuStreamBeginCapture", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT));
  private static final MethodHandle STREAM_END_CAPTURE = driver("cuStreamEndCapture", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS));
  private static final MethodHandle GRAPH_INSTANTIATE = driver("cuGraphInstantiateWithFlags", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG));
  private static final MethodHandle GRAPH_LAUNCH = driver("cuGraphLaunch", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG));
  private static final MethodHandle GRAPH_DESTROY = driver("cuGraphDestroy", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
  private static final MethodHandle MEMCPY_H2D_ASYNC = driver("cuMemcpyHtoDAsync", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_LONG));
  private static final MethodHandle MEMCPY_D2D_ASYNC =
      driver("cuMemcpyDtoDAsync",
          FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG));

  private static SymbolLookup driverLookup() {
    String explicit = System.getenv("CUDA_DRIVER_PATH");
    if (explicit != null && !explicit.isBlank()) {
      return SymbolLookup.libraryLookup(explicit, LIB_ARENA);
    }
    return SymbolLookup.libraryLookup("libcuda.so.1", LIB_ARENA);
  }

  private static MethodHandle driver(String name, FunctionDescriptor descriptor) {
    return LINKER.downcallHandle(CUDA.find(name).orElseThrow(() -> new UnsatisfiedLinkError(name)), descriptor);
  }

  // ---- NVRTC --------------------------------------------------------------

  private static final SymbolLookup NVRTC = nvrtcLookup();
  private static final MethodHandle NVRTC_CREATE = nvrtc("nvrtcCreateProgram", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle NVRTC_COMPILE = nvrtc("nvrtcCompileProgram", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
  private static final MethodHandle NVRTC_PTX_SIZE = nvrtc("nvrtcGetPTXSize", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle NVRTC_PTX = nvrtc("nvrtcGetPTX", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle NVRTC_LOG_SIZE = nvrtc("nvrtcGetProgramLogSize", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle NVRTC_LOG = nvrtc("nvrtcGetProgramLog", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

  private static SymbolLookup nvrtcLookup() {
    String explicit = System.getenv("NVRTC_PATH");
    if (explicit != null && !explicit.isBlank()) {
      return SymbolLookup.libraryLookup(explicit, LIB_ARENA);
    }
    String[] candidates = {
        "/lib/x86_64-linux-gnu/libnvrtc.so.11.5.119",
        "/usr/lib/x86_64-linux-gnu/libnvrtc.so.11.5.119"
    };
    for (String path : candidates) {
      if (Files.isRegularFile(Path.of(path))) {
        return SymbolLookup.libraryLookup(path, LIB_ARENA);
      }
    }
    return SymbolLookup.libraryLookup("libnvrtc.so", LIB_ARENA);
  }

  private static MethodHandle nvrtc(String name, FunctionDescriptor descriptor) {
    return LINKER.downcallHandle(NVRTC.find(name).orElseThrow(() -> new UnsatisfiedLinkError(name)), descriptor);
  }

  // ---- error handling -----------------------------------------------------

  private static void check(int status, String operation) {
    if (status == 0) {
      return;
    }
    String errorName = "?";
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment name = arena.allocate(ADDRESS);
      ERROR_NAME.invoke(status, name);
      errorName = name.get(ADDRESS, 0).reinterpret(128).getString(0);
    } catch (Throwable ignored) {
      // fall through with numeric code only
    }
    throw new RuntimeException(operation + " failed: " + errorName + " (code " + status + ")");
  }

  // ---- lifecycle ----------------------------------------------------------

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

  record DeviceInfo(int device, String name, int major, int minor) {
    String arch() {
      return "compute_" + major + minor;
    }
  }

  static DeviceInfo initContext(int ordinal) {
    try (Arena arena = Arena.ofConfined()) {
      check((int) INIT.invoke(0), "cuInit");
      MemorySegment deviceOut = arena.allocate(JAVA_INT);
      check((int) DEVICE_GET.invoke(deviceOut, ordinal), "cuDeviceGet");
      int device = deviceOut.get(JAVA_INT, 0);
      MemorySegment contextOut = arena.allocate(ADDRESS);
      check((int) CTX_CREATE.invoke(contextOut, 0, device), "cuCtxCreate");
      MemorySegment nameOut = arena.allocate(256);
      check((int) DEVICE_NAME.invoke(nameOut, 256, device), "cuDeviceGetName");
      String name = nameOut.getString(0);
      MemorySegment attr = arena.allocate(JAVA_INT);
      check((int) DEVICE_ATTRIBUTE.invoke(attr, 75, device), "cuDeviceGetAttribute(major)");
      int major = attr.get(JAVA_INT, 0);
      check((int) DEVICE_ATTRIBUTE.invoke(attr, 76, device), "cuDeviceGetAttribute(minor)");
      int minor = attr.get(JAVA_INT, 0);
      deviceTotalBytes = memGetInfo()[1];
      return new DeviceInfo(device, name, major, minor);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  /** Raw {@code cuDeviceGetAttribute} on the current device. */
  static int deviceAttribute(int attribute) {
    DeviceInfo info = context();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_INT);
      check((int) DEVICE_ATTRIBUTE.invoke(out, attribute, info.device()),
          "cuDeviceGetAttribute(" + attribute + ")");
      return out.get(JAVA_INT, 0);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static DeviceInfo sharedContext;

  /**
   * Lazily creates the one process-wide driver context. Every entry point that
   * needs the GPU goes through here so that a program mixing tensor work and
   * jit-compiled kernels does not end up with two contexts fighting over which
   * one is current on the calling thread.
   */
  static synchronized DeviceInfo context() {
    if (sharedContext == null) {
      sharedContext = initContext(0);
      createStream();
    }
    return sharedContext;
  }

  static void synchronize() {
    try {
      check((int) CTX_SYNC.invoke(), "cuCtxSynchronize");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  // ---- memory -------------------------------------------------------------

  private static final int CUDA_ERROR_OUT_OF_MEMORY = 2;
  private static final int OOM_RETRY_ATTEMPTS = 6;

  /**
   * Allocates device memory. A CudaBuffer wrapper is a few bytes on the Java
   * heap no matter how much device memory it holds, so the JVM's own GC
   * heuristics never feel pressured to collect them as device memory fills
   * up. Library code deliberately never calls {@code System.gc()} itself
   * here (a library forcing a full GC pause is an anti-pattern - it stalls
   * the whole JVM, including unrelated application code, on the library's
   * behalf). {@link #enforceBudget} still BLOCKS new allocations, before
   * they're attempted, whenever our own live buffers would cross a fraction
   * of total device memory - but it only waits for {@link #drainPendingFrees}
   * to catch up on GCs/Cleaner runs that already happened, it doesn't trigger
   * one. Applications whose workload leaves many intermediate {@code Tensor}s
   * unclosed under sustained load (instead of explicit {@code close()}/
   * try-with-resources) are responsible for hinting the JVM themselves - e.g.
   * a notebook driving nablatensor through JPype can call
   * {@code jpype.JClass("java.lang.System").gc()} periodically in its own
   * training/stress loop (see {@code notebooks/01_nablatensor_speedup.ipynb}'s
   * memory-leak stress test). The retry-on-OOM loop below remains as a
   * backstop for genuine external memory pressure, draining whatever the
   * Cleaner has already enqueued between attempts.
   */
  private static volatile long deviceTotalBytes = -1;
  private static final double HIGH_WATER_FRACTION = 0.5;
  private static final long BUDGET_MAX_WAIT_MILLIS = 2000;
  private static final long BUDGET_POLL_MILLIS = 5;

  static long malloc(long bytes) {
    drainPendingFrees();
    long recycled = takeFromPool(bytes);
    if (recycled != 0) {
      return recycled;
    }
    enforceBudget(bytes);
    for (int attempt = 1; attempt <= OOM_RETRY_ATTEMPTS; attempt++) {
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment out = arena.allocate(JAVA_LONG);
        int status = (int) MEM_ALLOC.invoke(out, bytes);
        if (status == 0) {
          return out.get(JAVA_LONG, 0);
        }
        if (status != CUDA_ERROR_OUT_OF_MEMORY || attempt == OOM_RETRY_ATTEMPTS) {
          if (status == CUDA_ERROR_OUT_OF_MEMORY) {
            long[] memory = memGetInfo();
            throw new RuntimeException("cuMemAlloc OOM: requested=" + formatMiB(bytes)
                + ", driverFree=" + formatMiB(memory[0])
                + ", driverTotal=" + formatMiB(memory[1])
                + ", live=" + formatMiB(CudaBuffer.LIVE_BYTES.get())
                + ", pooled=" + formatMiB(pooledBytes));
          }
          check(status, "cuMemAlloc");
        }
      } catch (Throwable failure) {
        throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
      }
      drainPendingFrees();
      flushPool(); // pooled blocks are real VRAM the driver still counts as in use
      try {
        Thread.sleep(20L * attempt);
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        throw new RuntimeException("interrupted while retrying cuMemAlloc after OOM", interrupted);
      }
    }
    throw new IllegalStateException("unreachable");
  }

  private static String formatMiB(long bytes) {
    return String.format(java.util.Locale.ROOT, "%.2f MiB", bytes / 1048576.0);
  }

  /**
   * Size-bucketed device-memory free list. {@code cuMemAlloc}/{@code cuMemFree}
   * are slow, implicitly-synchronizing driver calls, and a scenario-replay run
   * allocates the same handful of buffer sizes many times per path block,
   * so recycling pointers instead of handing them back to the driver removes
   * nearly all of that cost. Pooled blocks remain allocated to this process, so
   * the pool is capped and fully flushed before any OOM retry.
   */
  private static final Object POOL_LOCK = new Object();
  private static final java.util.Map<Long, java.util.ArrayDeque<Long>> FREE_POOL = new java.util.HashMap<>();
  private static final long POOL_MAX_BYTES = 256L * 1024 * 1024;
  private static long pooledBytes;

  private static long takeFromPool(long bytes) {
    synchronized (POOL_LOCK) {
      java.util.ArrayDeque<Long> bucket = FREE_POOL.get(bytes);
      if (bucket == null || bucket.isEmpty()) {
        return 0;
      }
      pooledBytes -= bytes;
      return bucket.removeLast();
    }
  }

  private static void recycle(long pointer, long bytes) {
    synchronized (POOL_LOCK) {
      if (pooledBytes + bytes <= POOL_MAX_BYTES) {
        FREE_POOL.computeIfAbsent(bytes, key -> new java.util.ArrayDeque<>()).addLast(pointer);
        pooledBytes += bytes;
        return;
      }
    }
    free(pointer);
  }

  private static void flushPool() {
    synchronized (POOL_LOCK) {
      for (java.util.ArrayDeque<Long> bucket : FREE_POOL.values()) {
        for (long pointer : bucket) {
          free(pointer);
        }
        bucket.clear();
      }
      pooledBytes = 0;
    }
  }

  /**
   * Blocks until our own live buffers (plus the pending request) fit under
   * the budget - this only waits/polls {@link #drainPendingFrees}, it does
   * not itself force a GC cycle (see {@link #malloc} javadoc).
   *
   * <p>Bails out as soon as a poll makes no progress (no bytes actually
   * freed), rather than always burning the full {@link #BUDGET_MAX_WAIT_MILLIS}.
   * Once the run's own long-lived resident buffers alone exceed the
   * high-water mark (held for the whole replay, never freed - e.g. a wide
   * path buffer plus the compiled tape's constants on a small card), every
   * single allocation for the rest of the run would otherwise pay a full
   * {@code BUDGET_MAX_WAIT_MILLIS} stall for a free that can never happen -
   * across the many kernel launches per path block that turns a one-off,
   * legitimate "let the Cleaner catch up" wait into an effective hang.
   * No-progress-after-one-poll means "nothing is pending", so waiting
   * longer cannot help; the OOM-retry loop in {@link #malloc} remains the
   * real backstop if the allocation genuinely doesn't fit.
   */
  private static void enforceBudget(long bytes) {
    if (deviceTotalBytes <= 0) {
      return;
    }
    long budget = (long) (deviceTotalBytes * HIGH_WATER_FRACTION);
    long live = CudaBuffer.LIVE_BYTES.get();
    if (live + bytes <= budget) {
      return;
    }
    long deadline = System.currentTimeMillis() + BUDGET_MAX_WAIT_MILLIS;
    while (live + bytes > budget && System.currentTimeMillis() < deadline) {
      drainPendingFrees();
      long after = CudaBuffer.LIVE_BYTES.get();
      if (after >= live) {
        break; // no bytes reclaimed this poll - further waiting won't change that
      }
      live = after;
      try {
        Thread.sleep(BUDGET_POLL_MILLIS);
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        return;
      }
    }
    drainPendingFrees();
  }

  private static long[] memGetInfo() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment freeOut = arena.allocate(JAVA_LONG);
      MemorySegment totalOut = arena.allocate(JAVA_LONG);
      check((int) MEM_GET_INFO.invoke(freeOut, totalOut), "cuMemGetInfo");
      return new long[] {freeOut.get(JAVA_LONG, 0), totalOut.get(JAVA_LONG, 0)};
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static void free(long pointer) {
    try {
      check((int) MEM_FREE.invoke(pointer), "cuMemFree");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  /**
   * CUDA contexts are thread-affine ({@code cuCtxCreate} only makes the context
   * current on the calling thread), but {@link java.lang.ref.Cleaner} cleaning
   * actions run on the Cleaner's own background thread - calling {@code cuMemFree}
   * directly from there has no context current and fails. So the Cleaner action
   * only enqueues here (thread-safe, no driver calls); the actual free happens
   * later via {@link #drainPendingFrees()}, called from the real CUDA thread.
   */
  private record PendingFree(long pointer, long bytes) {
  }

  private static final java.util.concurrent.ConcurrentLinkedQueue<PendingFree> PENDING_FREES = new java.util.concurrent.ConcurrentLinkedQueue<>();

  static void enqueueFree(long pointer, long bytes) {
    PENDING_FREES.add(new PendingFree(pointer, bytes));
  }

  static void drainPendingFrees() {
    PendingFree pending;
    while ((pending = PENDING_FREES.poll()) != null) {
      recycle(pending.pointer(), pending.bytes());
      CudaBuffer.LIVE_BYTES.addAndGet(-pending.bytes());
    }
  }

  static long uploadFloats(float[] data) {
    long bytes = (long) data.length * Float.BYTES;
    long pointer = malloc(bytes);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment host = arena.allocateFrom(JAVA_FLOAT, data);
      check((int) MEMCPY_H2D.invoke(pointer, host, bytes), "cuMemcpyHtoD");
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
      check((int) MEMCPY_D2H.invoke(host, pointer, bytes), "cuMemcpyDtoH");
      MemorySegment.copy(host, JAVA_FLOAT, 0, out, 0, count);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
    return out;
  }

  static void copyDeviceAsync(long target, long source, long bytes) {
    try {
      check((int) MEMCPY_D2D_ASYNC.invoke(target, source, bytes, stream), "cuMemcpyDtoDAsync");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static void uploadDoubles(long pointer, double[] data) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment host = arena.allocateFrom(JAVA_DOUBLE, data);
      check((int) MEMCPY_H2D.invoke(pointer, host, (long) data.length * Double.BYTES), "cuMemcpyHtoD");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static double[] downloadDoubles(long pointer, int count) {
    long bytes = (long) count * Double.BYTES;
    double[] out = new double[count];
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment host = arena.allocate(bytes, JAVA_DOUBLE.byteSize());
      check((int) MEMCPY_D2H.invoke(host, pointer, bytes), "cuMemcpyDtoH");
      MemorySegment.copy(host, JAVA_DOUBLE, 0, out, 0, count);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
    return out;
  }

  /** Uploads an opaque raw byte blob. */
  static long uploadBytes(byte[] data) {
    long bytes = data.length;
    long pointer = malloc(bytes);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment host = arena.allocate(bytes);
      MemorySegment.copy(data, 0, host, JAVA_BYTE, 0, data.length);
      check((int) MEMCPY_H2D.invoke(pointer, host, bytes), "cuMemcpyHtoD");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
    return pointer;
  }

  /** Uploads an opaque raw byte blob directly from an off-heap {@link MemorySegment}. */
  static long uploadSegment(MemorySegment source) {
    long bytes = source.byteSize();
    long pointer = malloc(bytes);
    try {
      check((int) MEMCPY_H2D.invoke(pointer, source, bytes), "cuMemcpyHtoD");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
    return pointer;
  }

  // ---- kernels ------------------------------------------------------------

  static byte[] compile(String source, String architecture) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment programOut = arena.allocate(ADDRESS);
      check((int) NVRTC_CREATE.invoke(programOut, arena.allocateFrom(source),
          arena.allocateFrom("nablatensor_kernels.cu"), 0, MemorySegment.NULL, MemorySegment.NULL), "nvrtcCreateProgram");
      MemorySegment program = programOut.get(ADDRESS, 0);
      String[] options = {"--gpu-architecture=" + architecture};
      MemorySegment optionPointers = arena.allocate(ADDRESS.byteSize() * options.length);
      for (int i = 0; i < options.length; i++) {
        optionPointers.setAtIndex(ADDRESS, i, arena.allocateFrom(options[i]));
      }
      int status = (int) NVRTC_COMPILE.invoke(program, options.length, optionPointers);
      if (status != 0) {
        throw new RuntimeException("nvrtcCompileProgram failed:\n" + programLog(program, arena));
      }
      MemorySegment size = arena.allocate(JAVA_LONG);
      check((int) NVRTC_PTX_SIZE.invoke(program, size), "nvrtcGetPTXSize");
      long length = size.get(JAVA_LONG, 0);
      MemorySegment buffer = arena.allocate(length);
      check((int) NVRTC_PTX.invoke(program, buffer), "nvrtcGetPTX");
      byte[] ptx = new byte[Math.toIntExact(length)];
      MemorySegment.copy(buffer, JAVA_BYTE, 0, ptx, 0, ptx.length);
      return ptx;
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static String programLog(MemorySegment program, Arena arena) throws Throwable {
    MemorySegment size = arena.allocate(JAVA_LONG);
    if ((int) NVRTC_LOG_SIZE.invoke(program, size) != 0) {
      return "";
    }
    long length = size.get(JAVA_LONG, 0);
    if (length <= 0) {
      return "";
    }
    MemorySegment buffer = arena.allocate(length + 1);
    NVRTC_LOG.invoke(program, buffer);
    return buffer.getString(0);
  }

  static long loadFunction(byte[] ptx, String kernelName) {
    try (Arena arena = Arena.ofConfined()) {
      if (System.getenv("NABLATENSOR_DUMP_PTX") != null) {
        java.nio.file.Files.write(java.nio.file.Path.of("/tmp/nablatensor_kernels.ptx"), ptx);
      }
      MemorySegment image = arena.allocate(ptx.length + 1);
      MemorySegment.copy(ptx, 0, image, JAVA_BYTE, 0, ptx.length);
      MemorySegment module = arena.allocate(JAVA_LONG);
      check((int) MODULE_LOAD.invoke(module, image), "cuModuleLoadData");
      MemorySegment function = arena.allocate(JAVA_LONG);
      check((int) MODULE_FUNCTION.invoke(function, module.get(JAVA_LONG, 0), arena.allocateFrom(kernelName)),
          "cuModuleGetFunction");
      return function.get(JAVA_LONG, 0);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  /** Launch a 1-D grid. Arguments are Long (device pointers), Integer, or Float. */
  static void launch(long function, int grid, int block, Object... arguments) {
    launchShared(function, grid, block, 0, arguments);
  }

  /** Launch a 1-D grid whose blocks use two thread dimensions. */
  static void launchBlocks2d(
      long function, int grid, int blockX, int blockY, Object... arguments) {
    prepareLaunchArguments(arguments);
    try {
      check((int) LAUNCH.invoke(function, grid, 1, 1, blockX, blockY, 1, 0, stream,
          LAUNCH_PARAMETERS, MemorySegment.NULL), "cuLaunchKernel");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  /**
   * Argument scratch reused across launches. A replay run issues many kernel
   * launches per path block, and allocating two confined {@link Arena} segments per
   * launch showed up as real host-side gap between kernels. The CUDA context is
   * thread-affine, so only the one CUDA thread ever touches these; the pointer
   * array always points at fixed offsets of the value array, so it is filled
   * once here rather than rebuilt per call.
   */
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

  static void launchShared(long function, int grid, int block, int sharedBytes, Object... arguments) {
    prepareLaunchArguments(arguments);
    try {
      check((int) LAUNCH.invoke(function, grid, 1, 1, block, 1, 1, sharedBytes, stream,
          LAUNCH_PARAMETERS, MemorySegment.NULL), "cuLaunchKernel");
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

  // ---- stream + graph capture ---------------------------------------------

  /**
   * Work is issued on a dedicated non-blocking stream rather than the legacy
   * default stream, because only a non-default stream can be captured into a
   * CUDA graph. A replay run issues many kernels per path block, and at a
   * few microseconds of host time each that launch cost was leaving the GPU
   * idle roughly a third of the time; replaying a captured graph collapses it
   * to a single launch.
   */
  private static volatile long stream;

  static void createStream() {
    if (stream != 0) {
      return;
    }
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_LONG);
      check((int) STREAM_CREATE.invoke(out, 1), "cuStreamCreate"); // CU_STREAM_NON_BLOCKING
      stream = out.get(JAVA_LONG, 0);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static void syncStream() {
    try {
      check((int) STREAM_SYNC.invoke(stream), "cuStreamSynchronize");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static void beginCapture() {
    try {
      check((int) STREAM_BEGIN_CAPTURE.invoke(stream, 0), "cuStreamBeginCapture"); // GLOBAL mode
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  /** Ends capture and returns an instantiated executable graph handle. */
  static long endCaptureAndInstantiate() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment graphOut = arena.allocate(JAVA_LONG);
      check((int) STREAM_END_CAPTURE.invoke(stream, graphOut), "cuStreamEndCapture");
      long graph = graphOut.get(JAVA_LONG, 0);
      MemorySegment execOut = arena.allocate(JAVA_LONG);
      check((int) GRAPH_INSTANTIATE.invoke(execOut, graph, 0L), "cuGraphInstantiateWithFlags");
      check((int) GRAPH_DESTROY.invoke(graph), "cuGraphDestroy");
      return execOut.get(JAVA_LONG, 0);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static void launchGraph(long graphExec) {
    try {
      check((int) GRAPH_LAUNCH.invoke(graphExec, stream), "cuGraphLaunch");
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static void uploadIntsAsync(long pointer, int[] values) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment host = arena.allocateFrom(JAVA_INT, values);
      check((int) MEMCPY_H2D_ASYNC.invoke(pointer, host, (long) values.length * Integer.BYTES, stream),
          "cuMemcpyHtoDAsync");
      syncStream(); // host buffer is confined to this scope
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }
}
