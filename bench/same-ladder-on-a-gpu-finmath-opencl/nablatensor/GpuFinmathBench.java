import java.util.*;
import net.finmath.montecarlo.*;
import net.finmath.montecarlo.assetderivativevaluation.*;
import net.finmath.montecarlo.assetderivativevaluation.models.BlackScholesModel;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiable;
import net.finmath.montecarlo.automaticdifferentiation.backward.RandomVariableDifferentiableAADFactory;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.opencl.montecarlo.RandomVariableOpenCLFactory;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.*;

/**
 * finmath-lib-opencl-extensions: the same European/Asian tapes as FinmathBench,
 * with RandomVariableOpenCLFactory in place of RandomVariableFromArrayFactory so
 * every RandomVariable op (and, for the AAD row, the operator graph built on top
 * of it) runs on the OpenCL device instead of a CPU array. One thread, as shipped
 * -- the library has no concept of sharding across GPU dispatches.
 *
 * The process does not exit on its own (JOCL keeps a non-daemon device executor
 * alive), hence the explicit System.exit at the end.
 */
public final class GpuFinmathBench {
  static final double S0 = 100, K = 100, VOL = 0.20, RATE = 0.03, T = 1.0;
  static final double[] SPOTS = {98, 99, 99.5, 100, 100.5, 101, 102};

  record Res(double price, double delta, double dK, double vega, double rho) {
    Res plus(Res o) { return new Res(price + o.price, delta + o.delta, dK + o.dK, vega + o.vega, rho + o.rho); }
  }

  public static void main(String[] a) throws Exception {
    String product = System.getProperty("product", "european");
    boolean greeks = Boolean.getBoolean("greeks");
    boolean cached = Boolean.getBoolean("cached");
    int paths = Integer.getInteger("paths", 20_000);
    int ladders = Integer.getInteger("ladders", 3);
    int steps = "european".equals(product) ? 1 : 252;
    TimeDiscretization td = new TimeDiscretizationFromArray(0.0, steps, T / steps);
    RandomVariableFactory plain = new RandomVariableOpenCLFactory();

    // warm-up on a different seed
    for (int w = 0; w < 2; w++) ladder(paths, td, steps, plain, greeks, cached, 7_000, 1);

    double[] cold = new double[ladders], warm = new double[ladders];
    Res at100 = null;
    for (int L = 0; L < ladders; L++) {
      List<double[]> r = ladder(paths, td, steps, plain, greeks, cached, 42 + 100 * L, SPOTS.length);
      double[] rate = r.get(0);
      at100 = new Res(r.get(1)[0], r.get(1)[1], r.get(1)[2], r.get(1)[3], r.get(1)[4]);
      cold[L] = rate[0];
      warm[L] = Arrays.stream(rate, 1, rate.length).average().orElse(0);
      System.out.printf(Locale.ROOT, "  ladder %d: cold %.4f warm %s%n", L, rate[0], Arrays.toString(Arrays.copyOfRange(rate, 1, rate.length)));
    }
    Arrays.sort(cold); Arrays.sort(warm);
    System.out.printf(Locale.ROOT,
        "RESULT lib=finmath-opencl product=%s steps=%d paths=%,d pass=%s cached=%s | cold %.4f Mpath/s | warm %.4f Mpath/s | price@100 %.6f delta %.6f dK %.6f vega %.6f rho %.6f%n",
        product, steps, paths, greeks ? "value+4G" : "price", cached,
        cold[ladders / 2], warm[ladders / 2], at100.price, at100.delta, at100.dK, at100.vega, at100.rho);
    System.exit(0);
  }

  static List<double[]> ladder(int paths, TimeDiscretization td, int steps, RandomVariableFactory plain,
      boolean greeks, boolean cached, int seed, int nSpots) throws Exception {
    BrownianMotion shared = cached ? new BrownianMotionFromMersenneRandomNumbers(td, 1, paths, seed, plain) : null;
    double[] rate = new double[nSpots];
    double[] v100 = new double[5];
    for (int i = 0; i < nSpots; i++) {
      double spot = SPOTS[i];
      long t0 = System.nanoTime();
      BrownianMotion bm = cached ? shared : new BrownianMotionFromMersenneRandomNumbers(td, 1, paths, seed, plain);
      Res r = greeks ? aad(bm, spot, steps, td) : price(bm, spot, steps, td, plain);
      double sec = (System.nanoTime() - t0) / 1e9;
      rate[i] = (double) paths / sec / 1e6;
      if (spot == 100.0) v100 = new double[] {r.price, r.delta, r.dK, r.vega, r.rho};
    }
    return List.of(rate, v100);
  }

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
    RandomVariableDifferentiableAADFactory f = new RandomVariableDifferentiableAADFactory(new RandomVariableOpenCLFactory());
    RandomVariableDifferentiable rvS = f.createRandomVariable(spot), rvR = f.createRandomVariable(RATE),
        rvV = f.createRandomVariable(VOL), rvK = f.createRandomVariable(K);
    BlackScholesModel model = new BlackScholesModel(rvS, rvR, rvV, f);
    // DIAGNOSTIC: rebuild bm through the AAD factory itself rather than the plain
    // OpenCL one passed in, to test whether mixed-factory type-priority promotion
    // is what's failing.
    BrownianMotion aadBm = new BrownianMotionFromMersenneRandomNumbers(td, 1, bm.getNumberOfPaths(), 123, f);
    AssetModelMonteCarloSimulationModel sim = new MonteCarloAssetModel(model, new EulerSchemeFromProcessModel(model, aadBm));
    RandomVariable value = payoff(sim, rvK, steps, td);
    Map<Long, RandomVariable> g = ((RandomVariableDifferentiable) value).getGradient();
    return new Res(value.getAverage(), g.get(rvS.getID()).getAverage(), g.get(rvK.getID()).getAverage(),
        g.get(rvV.getID()).getAverage(), g.get(rvR.getID()).getAverage());
  }
}
