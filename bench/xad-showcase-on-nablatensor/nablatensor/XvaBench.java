import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.Nabla;
import com.nablatensor.engine.SDouble;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Simplified XVA (CVA) pricer: a 15-swap portfolio, 20 semi-annual time
 * buckets, 40 differentiable market inputs (20 zero rates, 10 hazard rates,
 * 10 vol points), 10,000 MC paths — reimplemented from
 * ad-benchmarks/src/xva.hpp against NablaTensor. One quirk of the upstream
 * benchmark is kept exactly as written: the per-bucket CVA discount factor
 * uses the ORIGINAL (undiffused) rate curve, not the diffused path state used
 * to price the swaps' exposure — see xva_compute_cva in the upstream header.
 */
public final class XvaBench {
  static final double[] RATE_TENORS = {0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
      10.0, 12.0, 15.0, 17.0, 20.0, 22.0, 25.0, 27.0, 28.0, 30.0};
  static final double[] HAZARD_TENORS = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  static final int NUM_RATES = 20, NUM_HAZARD = 10, NUM_VOLS = 10, NUM_RANDOMS = 40;
  static final int NUM_SWAPS = 15, PAYMENTS_PER_SWAP = 20, TIME_BUCKETS = 20;
  static final double DT = 0.5;

  record Swap(double notional, double fixedRate, int numPayments, double startTime,
              double freq, boolean isPayer) {}

  public static void main(String[] args) {
    String engine = System.getProperty("engine", "cpu-jit");
    int threads = Integer.getInteger("threads", 8);
    boolean cached = Boolean.parseBoolean(System.getProperty("cached", "true"));
    long paths = Long.getLong("paths", 10_000L);
    int warmup = Integer.getInteger("warmup", 2);
    int iters = Integer.getInteger("iters", 7);
    int inner = Integer.getInteger("inner", 1);
    long seed = Long.getLong("seed", 99999L);

    if (cached) System.setProperty("nablatensor.crn", "on");

    Swap[] portfolio = new Swap[NUM_SWAPS];
    for (int i = 0; i < NUM_SWAPS; i++) {
      portfolio[i] = new Swap(1_000_000.0, 0.020 + 0.001 * i, PAYMENTS_PER_SWAP, 0.0, DT, i % 2 == 0);
    }

    Map<String, Double> base = new LinkedHashMap<>();
    for (int i = 0; i < NUM_RATES; i++) base.put("r" + i, 0.02 + 0.001 * i);
    for (int i = 0; i < NUM_HAZARD; i++) base.put("h" + i, 0.01 + 0.001 * i);
    for (int i = 0; i < NUM_VOLS; i++) base.put("vol" + i, 0.15 + 0.01 * i);

    try (Nabla.Pricer priced = Nabla.model(rec -> record(rec, portfolio))
        .fp64().greeks().threads(threads).on(engine).build();
        Nabla.Pricer valueOnly = Nabla.model(rec -> record(rec, portfolio))
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

      double sumRateGrad = 0, sumHazardGrad = 0, sumVolGrad = 0;
      for (int i = 0; i < NUM_RATES; i++) sumRateGrad += warm.greek("r" + i);
      for (int i = 0; i < NUM_HAZARD; i++) sumHazardGrad += warm.greek("h" + i);
      for (int i = 0; i < NUM_VOLS; i++) sumVolGrad += warm.greek("vol" + i);

      System.out.printf(Locale.ROOT,
          "RESULT bench=XVA-CVA engine=%s threads=%d cached=%s paths=%,d | primal %.4f ms | gradient %.4f ms | CVA %.4e%n",
          engine, threads, cached, paths, primal.medianMs(), grad.medianMs(), warm.price());
      System.out.printf(Locale.ROOT, "  sanity: CVA>=0 -> %s ; sum(dCVA/drate)=%.4e sum(dCVA/dhazard)=%.4e sum(dCVA/dvol)=%.4e%n",
          warm.price() >= 0.0, sumRateGrad, sumHazardGrad, sumVolGrad);
      // Note: ad-benchmarks/src/xva.hpp's additive-shock diffusion re-applies the SAME per-path
      // Gaussian shock at every one of the 20 time buckets (rates_out = rates_in + vol*sqrt(dt)*z,
      // z never redrawn), so a single moderately-unlucky draw compounds 20x into a curve point far
      // enough negative that exp(-rate*t) in the swap discount factor is astronomically large. This
      // is a property of the published toy model (confirmed with a standalone check outside NablaTensor
      // entirely), not a NablaTensor artifact; it does not affect the timing numbers, which measure the
      // same fixed amount of arithmetic regardless of the magnitude of the numbers flowing through it.
    }
  }

  static void record(AadRecorder rec, Swap[] portfolio) {
    SDouble[] rates = new SDouble[NUM_RATES];
    for (int i = 0; i < NUM_RATES; i++) rates[i] = rec.input("r" + i, 0.02 + 0.001 * i);
    SDouble[] hazard = new SDouble[NUM_HAZARD];
    for (int i = 0; i < NUM_HAZARD; i++) hazard[i] = rec.input("h" + i, 0.01 + 0.001 * i);
    SDouble[] vols = new SDouble[NUM_VOLS];
    for (int i = 0; i < NUM_VOLS; i++) vols[i] = rec.input("vol" + i, 0.15 + 0.01 * i);

    SDouble[] z = new SDouble[NUM_RANDOMS];
    for (int i = 0; i < NUM_RANDOMS; i++) z[i] = rec.randn();

    double sqrtDt = Math.sqrt(DT);
    SDouble[] diffused = rates.clone();
    SDouble cva = rec.constant(0.0);

    for (int bucket = 0; bucket < TIME_BUCKETS; bucket++) {
      double t = (bucket + 1) * DT;

      SDouble[] next = new SDouble[NUM_RATES];
      for (int i = 0; i < NUM_RATES; i++) {
        SDouble shock = vols[i % NUM_VOLS].mul(sqrtDt).mul(z[i % NUM_RANDOMS]);
        next[i] = diffused[i].add(shock);
      }
      diffused = next;

      SDouble portfolioPv = rec.constant(0.0);
      for (Swap swap : portfolio) {
        int paymentsElapsed = (int) (t / swap.freq());
        int remainingPayments = swap.numPayments() - paymentsElapsed;
        if (remainingPayments > 0) {
          Swap remaining = new Swap(swap.notional(), swap.fixedRate(), remainingPayments, t, swap.freq(), swap.isPayer());
          portfolioPv = portfolioPv.add(priceSwap(diffused, remaining));
        }
      }

      SDouble exposure = portfolioPv.max(0.0);
      SDouble survPrev = survivalProb(hazard, t - DT);
      SDouble survCurr = survivalProb(hazard, t);
      SDouble defaultProb = survPrev.sub(survCurr);
      SDouble df = discountFactor(rates, t); // original (undiffused) curve, matching upstream

      cva = cva.add(exposure.mul(defaultProb).mul(df));
    }
    rec.output(cva);
  }

  static SDouble priceSwap(SDouble[] rates, Swap swap) {
    SDouble fixedLeg = rates[0].mul(0.0); // typed zero
    SDouble floatLeg = fixedLeg;
    for (int i = 0; i < swap.numPayments(); i++) {
      double t = swap.startTime() + (i + 1) * swap.freq();
      SDouble df = discountFactor(rates, t);
      fixedLeg = fixedLeg.add(df.mul(swap.notional() * swap.fixedRate() * swap.freq()));

      double tPrev = swap.startTime() + i * swap.freq();
      SDouble dfPrev = discountFactor(rates, tPrev);
      SDouble fwdRate = dfPrev.div(df).sub(1.0).div(swap.freq());
      floatLeg = floatLeg.add(fwdRate.mul(df).mul(swap.notional() * swap.freq()));
    }
    return swap.isPayer() ? floatLeg.sub(fixedLeg) : fixedLeg.sub(floatLeg);
  }

  static SDouble discountFactor(SDouble[] rates, double t) {
    return interp(rates, RATE_TENORS, t).mul(-t).exp();
  }

  static SDouble survivalProb(SDouble[] hazard, double t) {
    if (t <= 0.0) return hazard[0].mul(0.0).add(1.0);
    return interp(hazard, HAZARD_TENORS, t).mul(-t).exp();
  }

  /** Linear interpolation with flat extrapolation, matching xva_discount_factor / xva_survival_prob. */
  static SDouble interp(SDouble[] arr, double[] tenors, double t) {
    int n = tenors.length;
    if (t <= tenors[0]) return arr[0];
    if (t >= tenors[n - 1]) return arr[n - 1];
    int idx = 0;
    for (int i = 0; i < n - 1; i++) {
      if (t < tenors[i + 1]) {
        idx = i;
        break;
      }
    }
    double w = (t - tenors[idx]) / (tenors[idx + 1] - tenors[idx]);
    return arr[idx].mul(1.0 - w).add(arr[idx + 1].mul(w));
  }
}
