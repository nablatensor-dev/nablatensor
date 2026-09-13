import java.util.Arrays;
import java.util.function.DoubleSupplier;

/** Shared warmup-then-median timing loop, mirroring ad-benchmarks' own src/timing.hpp. */
final class Bench {
  private Bench() {}

  record Stats(double medianMs, double minMs) {}

  /** Runs {@code fn} {@code innerLoop} times per sample after {@code warmup} untimed samples,
   * returns the median of {@code iters} timed samples (each sample's time divided by innerLoop). */
  static Stats time(Runnable fn, int warmup, int iters, int innerLoop) {
    for (int i = 0; i < warmup; i++) {
      for (int j = 0; j < innerLoop; j++) fn.run();
    }
    double[] times = new double[iters];
    for (int i = 0; i < iters; i++) {
      long t0 = System.nanoTime();
      for (int j = 0; j < innerLoop; j++) fn.run();
      long t1 = System.nanoTime();
      times[i] = (t1 - t0) / 1e6 / innerLoop;
    }
    Arrays.sort(times);
    return new Stats(times[times.length / 2], times[0]);
  }
}
