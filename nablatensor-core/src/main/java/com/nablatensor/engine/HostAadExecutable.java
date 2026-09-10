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
package com.nablatensor.engine;

import com.nablatensor.annotation.Internal;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * The half of an engine that a host (CPU) replay shares with every other host
 * replay: a worker pool, and the split of a path range across those workers.
 *
 * <p>Scenarios are the only parallel axis available — the tape is a chain of
 * dependent scalar operations, so it cannot be split — which means every host
 * engine wants exactly the same thing: contiguous, disjoint path ranges, one
 * per worker, each with its own scratch and its own {@link AadTotals}, summed
 * at the end. The scalar interpreter, the generated-bytecode kernel and the
 * vector sweeps differ only in what they do <em>inside</em> a range, so that is
 * all a subclass supplies. A subclass that walks the tape node by node reads it
 * through a {@link FlatTape}; the one that generates a kernel from it does not
 * need one.
 *
 * <p>The accelerator counterpart is {@link DeviceAadExecutable}: same base,
 * same chunking, but the parallelism is the device grid rather than a pool.
 *
 * <p>Public only so the engine modules can extend it; not part of the supported
 * API. Implement {@link AadEngine} / {@link AadExecutable} directly instead.
 */
@Internal
public abstract class HostAadExecutable extends AbstractAadExecutable {

  /** Resolved worker count; {@code 1} means the calling thread does the work. */
  protected final int threads;

  private final ExecutorService pool;

  protected HostAadExecutable(AadTape tape, AadOptions options, String threadName) {
    super(tape, options);
    this.threads = Math.max(1, options.resolvedThreads());
    this.pool = threads > 1
        ? Executors.newFixedThreadPool(threads, runnable -> {
            Thread thread = new Thread(runnable, threadName);
            thread.setDaemon(true);
            return thread;
          })
        : null;
  }

  /** Evaluates one contiguous path range on the calling thread. */
  @FunctionalInterface
  protected interface Range {
    AadTotals run(long pathFrom, long count);
  }

  /** Empty sums shaped for this tape. */
  protected final AadTotals newTotals() {
    return new AadTotals(tape.outputCount(), tape.inputCount());
  }

  /**
   * Runs {@code paths} scenarios from {@code pathOffset} across the workers and
   * sums their totals. {@code alignment} is the scenario granularity a worker
   * cannot split below — one for a scalar sweep, the batch width for a vector
   * one — so only the last worker can be handed a partial unit.
   */
  protected final AadTotals runRanges(long paths, long pathOffset, long alignment, Range range) {
    if (threads == 1) {
      return range.run(pathOffset, paths);
    }
    long units = (paths + alignment - 1) / alignment;
    long each = units / threads;
    long extra = units % threads;
    List<Callable<AadTotals>> tasks = new ArrayList<>(threads);
    long cursor = pathOffset;
    long remaining = paths;
    for (int t = 0; t < threads && remaining > 0; t++) {
      long count = Math.min((each + (t < extra ? 1 : 0)) * alignment, remaining);
      long from = cursor;
      cursor += count;
      remaining -= count;
      if (count > 0) {
        tasks.add(() -> range.run(from, count));
      }
    }
    AadTotals total = newTotals();
    try {
      for (Future<AadTotals> future : pool.invokeAll(tasks)) {
        total.add(future.get());
      }
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("replay interrupted", interrupted);
    } catch (ExecutionException failure) {
      Throwable cause = failure.getCause();
      throw cause instanceof RuntimeException runtime ? runtime : new IllegalStateException(cause);
    }
    return total;
  }

  /** Guards the common preconditions of a dispatch and returns its start time. */
  protected final long beginReplay(long paths) {
    checkOpen();
    if (paths <= 0) {
      throw new IllegalArgumentException("paths must be positive");
    }
    return System.nanoTime();
  }

  @Override
  public void close() {
    super.close();
    if (pool != null) {
      pool.shutdownNow();
    }
  }
}
