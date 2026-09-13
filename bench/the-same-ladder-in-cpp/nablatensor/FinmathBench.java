import java.util.*;
import java.util.concurrent.*;
import net.finmath.montecarlo.*;
import net.finmath.montecarlo.assetderivativevaluation.*;
import net.finmath.montecarlo.assetderivativevaluation.models.BlackScholesModel;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiable;
import net.finmath.montecarlo.automaticdifferentiation.backward.RandomVariableDifferentiableAADFactory;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.*;

/**
 * finmath-lib on the two tapes from the nablatensor articles:
 *   european: 1 step, T=1, S=K=100, vol 0.20, r 0.03  (price; price + delta, dV/dK, vega, rho via AAD)
 *   asian:    252 steps, arithmetic average of the 252 fixings, same market
 * Spot ladder 98..102 at a fixed seed; "cached" reuses one BrownianMotion (finmath memoises its
 * increments lazily), "fresh" builds a new BrownianMotion per call, which regenerates the draws.
 * -Dshards (default 1, as used in the article) optionally shards paths across threads with seed+shard each;
 * that is harness code, not the library, so the published numbers use shards=1. finmath itself is single-threaded per RandomVariable op.
 */
public final class FinmathBench {
  static final double S0 = 100, K = 100, VOL = 0.20, RATE = 0.03, T = 1.0;
  static final double[] SPOTS = {98, 99, 99.5, 100, 100.5, 101, 102};

  record Res(double price, double delta, double dK, double vega, double rho) {
    Res plus(Res o) { return new Res(price + o.price, delta + o.delta, dK + o.dK, vega + o.vega, rho + o.rho); }
    Res scale(double s) { return new Res(price * s, delta * s, dK * s, vega * s, rho * s); }
  }

  public static void main(String[] a) throws Exception {
    String product = System.getProperty("product", "european");
    boolean greeks = Boolean.getBoolean("greeks");
    boolean cached = Boolean.getBoolean("cached");
    int shards = Integer.getInteger("shards", 1);
    int paths = Integer.getInteger("paths", 8_000_000);
    int ladders = Integer.getInteger("ladders", 3);
    int steps = "european".equals(product) ? 1 : 252;
    TimeDiscretization td = new TimeDiscretizationFromArray(0.0, steps, T / steps);
    int perShard = paths / shards;
    ExecutorService pool = Executors.newFixedThreadPool(shards);
    RandomVariableFactory plain = new RandomVariableFromArrayFactory(true);

    // warm-up on a different seed
    for (int w = 0; w < 2; w++) ladder(pool, shards, perShard, td, steps, plain, greeks, cached, 7_000, 1).get(0);

    double[] cold = new double[ladders], warm = new double[ladders];
    Res at100 = null;
    for (int L = 0; L < ladders; L++) {
      List<double[]> r = ladder(pool, shards, perShard, td, steps, plain, greeks, cached, 42 + 100 * L, SPOTS.length);
      double[] rate = r.get(0);
      at100 = new Res(r.get(1)[0], r.get(1)[1], r.get(1)[2], r.get(1)[3], r.get(1)[4]);
      cold[L] = rate[0];
      warm[L] = Arrays.stream(rate, 1, rate.length).average().orElse(0);
      System.out.printf(Locale.ROOT, "  ladder %d: cold %.3f warm %s%n", L, rate[0], Arrays.toString(Arrays.copyOfRange(rate, 1, rate.length)));
    }
    Arrays.sort(cold); Arrays.sort(warm);
    System.out.printf(Locale.ROOT,
        "RESULT lib=finmath product=%s steps=%d paths=%,d shards=%d pass=%s cached=%s | cold %.3f Mpath/s | warm %.3f Mpath/s | price@100 %.6f delta %.6f dK %.6f vega %.6f rho %.6f%n",
        product, steps, perShard * shards, shards, greeks ? "value+4G" : "price", cached,
        cold[ladders / 2], warm[ladders / 2], at100.price, at100.delta, at100.dK, at100.vega, at100.rho);
    pool.shutdown();
  }

  /** returns [rates per spot], [price, delta, dK, vega, rho at spot 100]. */
  static List<double[]> ladder(ExecutorService pool, int shards, int perShard, TimeDiscretization td, int steps,
      RandomVariableFactory plain, boolean greeks, boolean cached, int seed, int nSpots) throws Exception {
    // one Brownian motion per shard, shared across the ladder when cached
    BrownianMotion[] shared = new BrownianMotion[shards];
    if (cached) for (int s = 0; s < shards; s++) shared[s] = new BrownianMotionFromMersenneRandomNumbers(td, 1, perShard, seed + s, plain);
    double[] rate = new double[nSpots];
    double[] v100 = new double[5];
    for (int i = 0; i < nSpots; i++) {
      final double spot = SPOTS[i];
      long t0 = System.nanoTime();
      List<Future<Res>> fs = new ArrayList<>();
      for (int s = 0; s < shards; s++) {
        final int sh = s;
        fs.add(pool.submit(() -> {
          BrownianMotion bm = cached ? shared[sh] : new BrownianMotionFromMersenneRandomNumbers(td, 1, perShard, seed + sh, plain);
          return greeks ? aad(bm, spot, steps, td) : price(bm, spot, steps, td, plain);
        }));
      }
      Res total = new Res(0, 0, 0, 0, 0);
      for (Future<Res> f : fs) total = total.plus(f.get());
      total = total.scale(1.0 / shards);
      double sec = (System.nanoTime() - t0) / 1e9;
      rate[i] = (double) perShard * shards / sec / 1e6;
      if (spot == 100.0) v100 = new double[] {total.price, total.delta, total.dK, total.vega, total.rho};
    }
    return List.of(rate, v100);
  }

  /** discounted payoff per path: European max(S_T - K, 0), Asian max(mean_i S_{t_i} - K, 0). */
  static RandomVariable payoff(AssetModelMonteCarloSimulationModel sim, RandomVariable strike, int steps, TimeDiscretization td) throws Exception {
    RandomVariable underlying;
    if (steps == 1) {
      underlying = sim.getAssetValue(T, 0);
    } else {
      RandomVariable sum = sim.getAssetValue(td.getTime(1), 0);
      for (int i = 2; i <= steps; i++) sum = sum.add(sim.getAssetValue(td.getTime(i), 0));
      underlying = sum.div(steps);
    }
    return underlying.sub(strike).floor(0.0).div(sim.getNumeraire(T)).mult(sim.getNumeraire(0.0));
  }

  static Res price(BrownianMotion bm, double spot, int steps, TimeDiscretization td, RandomVariableFactory f) throws Exception {
    BlackScholesModel model = new BlackScholesModel(f.createRandomVariable(spot), f.createRandomVariable(RATE), f.createRandomVariable(VOL), f);
    AssetModelMonteCarloSimulationModel sim = new MonteCarloAssetModel(model, new EulerSchemeFromProcessModel(model, bm));
    return new Res(payoff(sim, f.createRandomVariable(K), steps, td).getAverage(), 0, 0, 0, 0);
  }

  static Res aad(BrownianMotion bm, double spot, int steps, TimeDiscretization td) throws Exception {
    RandomVariableDifferentiableAADFactory f = new RandomVariableDifferentiableAADFactory(new RandomVariableFromArrayFactory(true));
    RandomVariableDifferentiable rvS = f.createRandomVariable(spot), rvR = f.createRandomVariable(RATE),
        rvV = f.createRandomVariable(VOL), rvK = f.createRandomVariable(K);
    BlackScholesModel model = new BlackScholesModel(rvS, rvR, rvV, f);
    AssetModelMonteCarloSimulationModel sim = new MonteCarloAssetModel(model, new EulerSchemeFromProcessModel(model, bm));
    RandomVariable value = payoff(sim, rvK, steps, td);
    Map<Long, RandomVariable> g = ((RandomVariableDifferentiable) value).getGradient();
    return new Res(value.getAverage(), g.get(rvS.getID()).getAverage(), g.get(rvK.getID()).getAverage(),
        g.get(rvV.getID()).getAverage(), g.get(rvR.getID()).getAverage());
  }
}
