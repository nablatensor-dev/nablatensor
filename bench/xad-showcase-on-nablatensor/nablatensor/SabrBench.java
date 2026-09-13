import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.Nabla;
import com.nablatensor.engine.SDouble;
import java.util.Locale;

/**
 * SABR (Hagan 2002) multi-expiry surface calibration objective, 15
 * differentiable inputs (alpha, rho, nu per expiry x 5 expiries), wrapped in
 * ad-benchmarks' fake 500-iteration "optimizer loop" (a fixed perturbation
 * schedule, not a real converging fit — see ad-benchmarks/src/sabr.hpp).
 *
 * <p>Unlike XAD's own harness, which re-records the tape every iteration
 * (tape.clearAll() + registerInput() + newRecording() 500 times), this builds
 * the NablaTensor kernel once and calls {@code request.with(...).run()} 500
 * times — one adjoint sweep per iteration either way, but no re-tracing,
 * because the recorded structure never changes between iterations. That is a
 * genuine NablaTensor property (build once, replay cheaply), not a shortcut
 * taken for this benchmark, and the article this harness supports says so.
 */
public final class SabrBench {
  static final int EXPIRIES = 5;
  static final int STRIKES = 20;
  static final int PARAMS = 3 * EXPIRIES;
  static final int CALIB_ITERS = 500;

  record Slice(double F, double expiry, double beta, double[] strikes, double[] marketVols) {}

  public static void main(String[] args) {
    String engine = System.getProperty("engine", "cpu-jit");
    int threads = Integer.getInteger("threads", 8);
    int warmup = Integer.getInteger("warmup", 3);
    int iters = Integer.getInteger("iters", 10);

    double[] expiries = {1.0, 2.0, 5.0, 10.0, 20.0};
    double[] forwards = {0.025, 0.028, 0.030, 0.032, 0.035};
    double[] trueAlpha = {0.040, 0.038, 0.035, 0.032, 0.030};
    double[] trueRho = {-0.15, -0.20, -0.25, -0.28, -0.30};
    double[] trueNu = {0.50, 0.45, 0.40, 0.35, 0.30};
    double[] initAlpha = new double[EXPIRIES], initRho = new double[EXPIRIES], initNu = new double[EXPIRIES];
    Slice[] slices = new Slice[EXPIRIES];
    for (int e = 0; e < EXPIRIES; e++) {
      initAlpha[e] = trueAlpha[e] + 0.005;
      initRho[e] = trueRho[e] + 0.10;
      initNu[e] = trueNu[e] - 0.05;
      double kMin = forwards[e] * 0.3, kMax = forwards[e] * 2.0;
      double[] strikes = new double[STRIKES];
      double[] marketVols = new double[STRIKES];
      for (int i = 0; i < STRIKES; i++) {
        strikes[i] = kMin + (kMax - kMin) * i / (STRIKES - 1);
        marketVols[i] = haganDouble(trueAlpha[e], 0.5, trueRho[e], trueNu[e], forwards[e], strikes[i], expiries[e]);
      }
      slices[e] = new Slice(forwards[e], expiries[e], 0.5, strikes, marketVols);
    }

    // sanity check: Hagan at the true params must reproduce the market vols it generated
    double maxErr = 0;
    for (int e = 0; e < EXPIRIES; e++) {
      for (int i = 0; i < STRIKES; i++) {
        double v = haganDouble(trueAlpha[e], 0.5, trueRho[e], trueNu[e], slices[e].F(), slices[e].strikes()[i], slices[e].expiry());
        maxErr = Math.max(maxErr, Math.abs(v - slices[e].marketVols()[i]));
      }
    }
    System.out.printf(Locale.ROOT, "  sanity: Hagan @ true params vs synthetic market vols, max abs err = %.3e%n", maxErr);

    double[][] dp = new double[CALIB_ITERS][PARAMS];
    for (int i = 0; i < CALIB_ITERS; i++) {
      double decay = 1.0 / (1.0 + 0.01 * i);
      double phase = 0.1 * i;
      for (int p = 0; p < PARAMS; p++) {
        double freq = 1.0 + 0.3 * p;
        double scale = (p % 3 == 0) ? 0.0002 : 0.001;
        dp[i][p] = scale * decay * Math.cos(phase * freq);
      }
    }

    try (Nabla.Pricer priced = Nabla.model(rec -> record(rec, slices)).fp64().greeks().threads(threads).on(engine).build();
        Nabla.Pricer valueOnly = Nabla.model(rec -> record(rec, slices)).fp64().priceOnly().threads(threads).on(engine).build()) {
      Nabla.Request req = priced.value();
      req.scenarios(1); // deterministic objective, no Monte Carlo: 1 scenario, not Nabla's 1M MC default
      Nabla.Request preq = valueOnly.value();
      preq.scenarios(1);
      double[] paramsD = new double[PARAMS];
      for (int e = 0; e < EXPIRIES; e++) {
        paramsD[3 * e] = initAlpha[e];
        paramsD[3 * e + 1] = initRho[e];
        paramsD[3 * e + 2] = initNu[e];
      }
      setInputs(req, paramsD);

      // finite-difference check on one gradient component before trusting timings
      Nabla.Valuation v0 = req.run();
      double adjointDAlpha0 = v0.greek("p0");
      double h = 1e-6;
      double[] bumped = paramsD.clone();
      bumped[0] += h;
      setInputs(req, bumped);
      double fUp = req.run().price();
      bumped[0] -= 2 * h;
      setInputs(req, bumped);
      double fDown = req.run().price();
      double fd = (fUp - fDown) / (2 * h);
      System.out.printf(Locale.ROOT, "  sanity: d(objective)/d(alpha_0) adjoint=%.6f  finite-diff=%.6f%n", adjointDAlpha0, fd);
      setInputs(req, paramsD);

      Runnable fullLoop = () -> {
        double[] params = new double[PARAMS];
        for (int e = 0; e < EXPIRIES; e++) {
          params[3 * e] = initAlpha[e];
          params[3 * e + 1] = initRho[e];
          params[3 * e + 2] = initNu[e];
        }
        double sink = 0.0;
        for (int iter = 0; iter < CALIB_ITERS; iter++) {
          setInputs(req, params);
          Nabla.Valuation v = req.run();
          sink += v.price();
          for (int p = 0; p < PARAMS; p++) {
            sink += v.greek("p" + p);
            params[p] += dp[iter][p];
          }
        }
        if (sink == Double.NaN) throw new AssertionError(); // keep sink live
      };

      Runnable primalLoop = () -> {
        double[] params = new double[PARAMS];
        for (int e = 0; e < EXPIRIES; e++) {
          params[3 * e] = initAlpha[e];
          params[3 * e + 1] = initRho[e];
          params[3 * e + 2] = initNu[e];
        }
        double sink = 0.0;
        for (int iter = 0; iter < CALIB_ITERS; iter++) {
          setInputs(preq, params);
          sink += preq.run().price();
          for (int p = 0; p < PARAMS; p++) params[p] += dp[iter][p];
        }
        if (sink == Double.NaN) throw new AssertionError();
      };
      Bench.Stats primal = Bench.time(primalLoop, warmup, iters, 1);
      Bench.Stats grad = Bench.time(fullLoop, warmup, iters, 1);
      System.out.printf(Locale.ROOT,
          "RESULT bench=SABRCalib engine=%s threads=%d iterations=%d | primal(full-loop) %.4f ms | gradient(full-loop) %.4f ms | objective@init %.6e%n",
          engine, threads, CALIB_ITERS, primal.medianMs(), grad.medianMs(), v0.price());
    }
  }

  static void setInputs(Nabla.Request req, double[] params) {
    for (int p = 0; p < PARAMS; p++) req.with("p" + p, params[p]);
  }

  static void record(AadRecorder rec, Slice[] slices) {
    // Record-time defaults must avoid alpha=0 (the Hagan formula divides by alpha);
    // use plausible SABR values per role (alpha, rho, nu repeating every 3 params).
    SDouble[] params = new SDouble[PARAMS];
    for (int p = 0; p < PARAMS; p++) {
      double dflt = switch (p % 3) {
        case 0 -> 0.035;  // alpha
        case 1 -> -0.2;   // rho
        default -> 0.4;   // nu
      };
      params[p] = rec.input("p" + p, dflt);
    }

    SDouble total = rec.constant(0.0);
    for (int e = 0; e < EXPIRIES; e++) {
      SDouble alpha = params[3 * e];
      SDouble rho = params[3 * e + 1];
      SDouble nu = params[3 * e + 2];
      Slice sl = slices[e];
      for (int i = 0; i < STRIKES; i++) {
        SDouble model = hagan(alpha, sl.beta(), rho, nu, sl.F(), sl.strikes()[i], sl.expiry());
        SDouble diff = model.sub(sl.marketVols()[i]);
        total = total.add(diff.mul(diff));
      }
    }
    rec.output(total);
  }

  /** Hagan 2002 SABR implied vol; F, K, beta, expiry are record-time constants, matching
   * ad-benchmarks/src/sabr.hpp's templated function where only alpha/rho/nu are the AD type. */
  static SDouble hagan(SDouble alpha, double beta, SDouble rho, SDouble nu, double F, double K, double expiry) {
    double oneMBeta = 1.0 - beta;
    double oneMBeta2 = oneMBeta * oneMBeta;
    double oneMBeta4 = oneMBeta2 * oneMBeta2;
    double fkPow = Math.pow(F * K, oneMBeta / 2.0);
    double fkPowFull = Math.pow(F * K, oneMBeta);
    double logFK = Math.log(F / K);
    double logFK2 = logFK * logFK;
    double logFK4 = logFK2 * logFK2;

    SDouble alpha2 = alpha.mul(alpha);
    SDouble correction = alpha2.div(fkPowFull).mul(oneMBeta2 / 24.0)
        .add(rho.mul(beta).mul(nu).mul(alpha).div(fkPow).mul(0.25))
        .add(rho.mul(rho).neg().mul(3.0).add(2.0).mul(nu).mul(nu).div(24.0))
        .mul(expiry).add(1.0);

    double atmTol = 1e-7 * F;
    if (Math.abs(F - K) < atmTol) {
      return alpha.div(Math.pow(F, oneMBeta)).mul(correction);
    }

    double denomGeo = fkPow * (1.0 + oneMBeta2 / 24.0 * logFK2 + oneMBeta4 / 1920.0 * logFK4);
    SDouble z = nu.div(alpha).mul(fkPow).mul(logFK);
    SDouble sqrtTerm = rho.mul(z).mul(2.0).neg().add(1.0).add(z.mul(z)).sqrt();
    SDouble xZ = sqrtTerm.add(z).sub(rho).div(rho.neg().add(1.0)).log();
    SDouble zOverXz = z.div(xZ);
    return alpha.div(denomGeo).mul(zOverXz).mul(correction);
  }

  static double haganDouble(double alpha, double beta, double rho, double nu, double F, double K, double expiry) {
    double oneMBeta = 1.0 - beta;
    double oneMBeta2 = oneMBeta * oneMBeta;
    double oneMBeta4 = oneMBeta2 * oneMBeta2;
    double fkPow = Math.pow(F * K, oneMBeta / 2.0);
    double fkPowFull = Math.pow(F * K, oneMBeta);
    double logFK = Math.log(F / K);
    double logFK2 = logFK * logFK;
    double logFK4 = logFK2 * logFK2;
    double alpha2 = alpha * alpha;
    double correction = 1.0 + (oneMBeta2 / 24.0 * alpha2 / fkPowFull
        + 0.25 * rho * beta * nu * alpha / fkPow
        + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu) * expiry;
    double atmTol = 1e-7 * F;
    if (Math.abs(F - K) < atmTol) {
      return alpha / Math.pow(F, oneMBeta) * correction;
    }
    double denomGeo = fkPow * (1.0 + oneMBeta2 / 24.0 * logFK2 + oneMBeta4 / 1920.0 * logFK4);
    double z = nu / alpha * fkPow * logFK;
    double sqrtTerm = Math.sqrt(1.0 - 2.0 * rho * z + z * z);
    double xZ = Math.log((sqrtTerm + z - rho) / (1.0 - rho));
    return alpha / denomGeo * (z / xZ) * correction;
  }
}
