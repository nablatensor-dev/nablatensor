import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.Nabla;
import com.nablatensor.engine.SDouble;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * LIBOR market model swaption portfolio, 161 differentiable inputs (1 delta +
 * 80 forward rates L0 + 80 vols lambda), 10,000 MC paths. Adapted from Mike
 * Giles' benchmark (people.maths.ox.ac.uk/~gilesm/codes/libor_AD/testlinadj.cpp)
 * via ad-benchmarks/src/libor_swaption.hpp, reimplemented against NablaTensor.
 * This is the famous "many more inputs than a bump-and-revalue could afford"
 * case (Giles &amp; Glasserman, "Smoking Adjoints") — the reverse-mode sweep must
 * hand back all 161 adjoints from one pass, not 161 separate one-at-a-time
 * gradients, which is the entire point of the benchmark existing.
 */
public final class LiborBench {
  static final int NN = 80, N = NN / 2, SAMPLES = NN / 2;
  static final int[] MATURITIES = {4, 4, 4, 8, 8, 8, 20, 20, 20, 28, 28, 28, 40, 40, 40};
  static final double[] SWAPRATES = {.045, .05, .055, .045, .05, .055, .045, .05, .055, .045, .05, .055, .045, .05, .055};

  public static void main(String[] args) {
    String engine = System.getProperty("engine", "cpu-jit");
    int threads = Integer.getInteger("threads", 8);
    boolean cached = Boolean.parseBoolean(System.getProperty("cached", "true"));
    long paths = Long.getLong("paths", 10_000L);
    int warmup = Integer.getInteger("warmup", 3);
    int iters = Integer.getInteger("iters", 10);
    int inner = Integer.getInteger("inner", 5);
    long seed = Long.getLong("seed", 12354L);

    if (cached) System.setProperty("nablatensor.crn", "on");

    Map<String, Double> base = new LinkedHashMap<>();
    base.put("delta", 0.05);
    for (int i = 0; i < NN; i++) base.put("L0_" + i, 0.05);
    for (int i = 0; i < NN; i++) base.put("lambda_" + i, 0.20);

    try (Nabla.Pricer priced = Nabla.model(LiborBench::record)
        .fp64().greeks().threads(threads).on(engine).build();
        Nabla.Pricer valueOnly = Nabla.model(LiborBench::record)
            .fp64().priceOnly().threads(1).on(engine).build()) {

      Nabla.Request p = valueOnly.value();
      base.forEach(p::with);
      p.scenarios(paths).seed(seed);
      Bench.Stats primal = Bench.time(p::run, warmup, iters, inner);

      Nabla.Request g = priced.value();
      base.forEach(g::with);
      g.scenarios(paths).seed(seed);
      Bench.time(g::run, warmup, 1, 1);
      Nabla.Valuation warm = g.run();
      Bench.Stats grad = Bench.time(g::run, 1, iters, inner);

      // finite-difference sanity check on 3 of the 161 adjoints, common random numbers
      // (same seed => same draws for bump up/down, so MC noise mostly cancels)
      double h = 1e-6;
      double fdDelta = bumpFd(priced, base, "delta", h, paths, seed);
      double fdL0 = bumpFd(priced, base, "L0_0", h, paths, seed);
      double fdLambda = bumpFd(priced, base, "lambda_0", h, paths, seed);
      System.out.printf(Locale.ROOT,
          "  sanity: d(price)/d(delta)   adjoint=%.6f  finite-diff=%.6f%n", warm.greek("delta"), fdDelta);
      System.out.printf(Locale.ROOT,
          "  sanity: d(price)/d(L0_0)    adjoint=%.6f  finite-diff=%.6f%n", warm.greek("L0_0"), fdL0);
      System.out.printf(Locale.ROOT,
          "  sanity: d(price)/d(lambda_0) adjoint=%.6f  finite-diff=%.6f%n", warm.greek("lambda_0"), fdLambda);

      double sumL0 = 0, sumLambda = 0;
      for (int i = 0; i < NN; i++) sumL0 += warm.greek("L0_" + i);
      for (int i = 0; i < NN; i++) sumLambda += warm.greek("lambda_" + i);

      System.out.printf(Locale.ROOT,
          "RESULT bench=LiborSwaption engine=%s threads=%d cached=%s paths=%,d inputs=161 | primal %.4f ms | gradient %.4f ms | price %.6f%n",
          engine, threads, cached, paths, primal.medianMs(), grad.medianMs(), warm.price());
      System.out.printf(Locale.ROOT, "  sum(dV/dL0)=%.4f sum(dV/dlambda)=%.4f dV/ddelta=%.4f%n",
          sumL0, sumLambda, warm.greek("delta"));
    }
  }

  static double bumpFd(Nabla.Pricer priced, Map<String, Double> base, String name, double h, long paths, long seed) {
    Nabla.Request up = priced.value();
    base.forEach(up::with);
    up.with(name, base.get(name) + h).scenarios(paths).seed(seed);
    double fUp = up.run().price();

    Nabla.Request down = priced.value();
    base.forEach(down::with);
    down.with(name, base.get(name) - h).scenarios(paths).seed(seed);
    double fDown = down.run().price();

    return (fUp - fDown) / (2 * h);
  }

  static void record(AadRecorder rec) {
    SDouble delta = rec.input("delta", 0.05);
    SDouble[] L = new SDouble[NN];
    SDouble[] lambda = new SDouble[NN];
    for (int i = 0; i < NN; i++) L[i] = rec.input("L0_" + i, 0.05);
    for (int i = 0; i < NN; i++) lambda[i] = rec.input("lambda_" + i, 0.20);

    SDouble[] z = new SDouble[SAMPLES];
    for (int n = 0; n < SAMPLES; n++) z[n] = rec.randn();

    SDouble sqrtDelta = delta.sqrt();

    for (int n = 0; n < SAMPLES; n++) {
      SDouble sqez = sqrtDelta.mul(z[n]);
      SDouble v = rec.constant(0.0);
      for (int i = n + 1; i < NN; i++) {
        SDouble lam = lambda[i - n - 1];
        SDouble con1 = delta.mul(lam);
        SDouble denom = L[i].mul(delta).add(1.0);
        SDouble term = con1.mul(L[i]).div(denom);
        v = v.add(term);
        SDouble growth = con1.mul(v).add(lam.mul(sqez.sub(con1.mul(0.5)))).exp();
        L[i] = L[i].mul(growth);
      }
    }

    SDouble b = rec.constant(1.0);
    SDouble s = rec.constant(0.0);
    SDouble[] btmp = new SDouble[NN];
    SDouble[] stmp = new SDouble[NN];
    for (int n = N; n < NN; n++) {
      b = b.div(L[n].mul(delta).add(1.0));
      s = s.add(b.mul(delta));
      btmp[n] = b;
      stmp[n] = s;
    }

    SDouble v = rec.constant(0.0);
    for (int i = 0; i < MATURITIES.length; i++) {
      int m = MATURITIES[i] + N - 1;
      SDouble swapval = btmp[m].add(stmp[m].mul(SWAPRATES[i])).sub(1.0);
      v = v.add(swapval.min(0.0).mul(-100.0));
    }
    for (int n = 0; n < N; n++) {
      v = v.div(L[n].mul(delta).add(1.0));
    }
    rec.output(v);
  }
}
