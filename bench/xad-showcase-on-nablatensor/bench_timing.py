"""Warmup-then-median timing, mirroring ad-benchmarks' own src/timing.hpp and this
project's Bench.java (bench/lib-comparison-adbench)."""
import statistics
import time


def time_it(fn, warmup=3, iters=10, inner=1):
    for _ in range(warmup):
        for _ in range(inner):
            fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return statistics.median(samples) * 1000.0  # ms
