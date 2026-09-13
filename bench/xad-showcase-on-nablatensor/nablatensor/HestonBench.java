import com.nablatensor.engine.Nabla;
import com.nablatensor.engine.SDouble;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Heston stochastic-vol Monte Carlo, 8 differentiable inputs, 100 Euler steps,
 * European call — the first of auto-differentiation/ad-benchmarks' four
 * scenarios, reimplemented against NablaTensor instead of XAD/CppAD/Adept/autodiff.
 * Same formula as ad-benchmarks/src/heston.hpp; RNG is NablaTensor's own Philox
 * stream, not XAD's mt19937, so bit-for-bit draws are not expected to match —
 * only the priced value and the Greeks' sign/magnitude are checked for sanity.
 */
public final class HestonBench {
  static final int STEPS = 100;
  static final double RHO = -0.7;

  public static void main(String[] args) {
    String engine = System.getProperty("engine", "cpu-jit");
    int threads = Integer.getInteger("threads", 8);
    boolean cached = Boolean.parseBoolean(System.getProperty("cached", "true"));
    long paths = Long.getLong("paths", 10_000L);
    int warmup = Integer.getInteger("warmup", 3);
    int iters = Integer.getInteger("iters", 20);
    int inner = Integer.getInteger("inner", 20);
    long seed = Long.getLong("seed", 77777L);

    if (cached) System.setProperty("nablatensor.crn", "on");

    Map<String, Double> base = new LinkedHashMap<>();
    base.put("S0", 100.0);
    base.put("K", 105.0);
    base.put("T", 1.0);
    base.put("r", 0.05);
    base.put("v0", 0.04);
    base.put("kappa", 2.0);
    base.put("theta", 0.04);
    base.put("xi", 0.3);

    try (Nabla.Pricer priced = Nabla.model(HestonBench::record)
        .fp64().greeks().threads(threads).on(engine).build();
        Nabla.Pricer valueOnly = Nabla.model(HestonBench::record)
            .fp64().priceOnly().threads(1).on(engine).build()) {

      // --- primal, 1 thread ---
      Nabla.Request p = valueOnly.value();
      base.forEach(p::with);
      p.scenarios(paths).seed(seed);
      Bench.Stats primal = Bench.time(p::run, warmup, iters, inner);

      // --- gradient ---
      Nabla.Request g = priced.value();
      base.forEach(g::with);
      g.scenarios(paths).seed(seed);
      Bench.time(g::run, warmup, 1, 1); // warm HotSpot on unrelated calls
      Nabla.Valuation warm = g.run();   // populate the draw cache for this seed
      Bench.Stats grad = Bench.time(g::run, 1, iters, inner);

      System.out.printf(Locale.ROOT,
          "RESULT bench=HestonMC engine=%s threads=%d cached=%s paths=%,d | primal %.4f ms | gradient %.4f ms | price %.6f%n",
          engine, threads, cached, paths, primal.medianMs(), grad.medianMs(), warm.price());
      System.out.printf(Locale.ROOT, "  greeks: dS0=%.6f dK=%.6f dT=%.6f dr=%.6f dv0=%.6f dkappa=%.6f dtheta=%.6f dxi=%.6f%n",
          warm.greek("S0"), warm.greek("K"), warm.greek("T"), warm.greek("r"),
          warm.greek("v0"), warm.greek("kappa"), warm.greek("theta"), warm.greek("xi"));
    }
  }

  static void record(com.nablatensor.engine.AadRecorder rec) {
    SDouble S0 = rec.input("S0", 100.0);
    SDouble K = rec.input("K", 105.0);
    SDouble T = rec.input("T", 1.0);
    SDouble r = rec.input("r", 0.05);
    SDouble v0 = rec.input("v0", 0.04);
    SDouble kappa = rec.input("kappa", 2.0);
    SDouble theta = rec.input("theta", 0.04);
    SDouble xi = rec.input("xi", 0.3);

    SDouble dt = T.div((double) STEPS);
    SDouble sqrtDt = dt.sqrt();
    double sqrtOneMinusRho2 = Math.sqrt(1.0 - RHO * RHO);

    SDouble S = S0;
    SDouble v = v0;
    for (int i = 0; i < STEPS; i++) {
      SDouble z1 = rec.randn();
      SDouble z2 = rec.randn();
      SDouble dW1 = z1;
      SDouble dW2 = z1.mul(RHO).add(z2.mul(sqrtOneMinusRho2));

      SDouble sqrtV = v.abs().add(1e-10).sqrt();

      SDouble sNext = S.add(r.mul(S).mul(dt)).add(sqrtV.mul(S).mul(sqrtDt).mul(dW1));
      SDouble vNext = v.add(kappa.mul(theta.sub(v)).mul(dt)).add(xi.mul(sqrtV).mul(sqrtDt).mul(dW2));
      S = sNext;
      v = vNext;
    }

    SDouble payoff = S.sub(K).max(0.0);
    SDouble disc = r.neg().mul(T).exp();
    rec.output(disc.mul(payoff));
  }
}
