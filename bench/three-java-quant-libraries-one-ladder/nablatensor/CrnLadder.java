import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Products;
import java.util.Arrays;
import java.util.Locale;

/** Spot-ladder harness: cold (first call, populates cache) vs warm (same seed, shocked spot). */
public final class CrnLadder {
  public static void main(String[] a) {
    String engine = System.getProperty("engine", "cpu-jit");
    String product = System.getProperty("product", "european");
    int steps = "european".equals(product) ? 1 : Integer.getInteger("steps", 252);
    long paths = Long.getLong("paths", 20_000_000L);
    boolean greeks = Boolean.getBoolean("greeks");
    int ladders = Integer.getInteger("ladders", 3);
    int threads = Integer.getInteger("threads", 8);
    double[] spots = {98, 99, 99.5, 100, 100.5, 101, 102};
    EquityMarket base = EquityMarket.atmOneYear();

    var b = MonteCarlo.of("european".equals(product) ? Products.europeanCall() : Products.asianCall())
        .market(base).steps(steps).fp64().threads(threads).on(engine);
    b = greeks ? b.greeks() : b.priceOnly();
    try (MonteCarlo<EquityMarket> mc = b.build()) {
      // warm HotSpot on a different seed, so the ladder's first call is a genuine cache miss
      for (int i = 0; i < 4; i++) mc.run(base, paths, 7L);
      double[] cold = new double[ladders];
      double[] warm = new double[ladders];
      double lastPrice = 0, lastDelta = 0;
      for (int L = 0; L < ladders; L++) {
        long seed = 42L + L;   // a fresh seed per ladder => first call of each ladder is a miss
        double[] rate = new double[spots.length];
        for (int i = 0; i < spots.length; i++) {
          var r = mc.run(base.withSpot(spots[i]), paths, seed);
          rate[i] = r.scenariosPerSecond() / 1e6;
          if (spots[i] == 100.0) { lastPrice = r.price(); lastDelta = greeks ? r.greek(EquityMarket::spot) : Double.NaN; }
        }
        cold[L] = rate[0];
        warm[L] = Arrays.stream(rate, 1, rate.length).average().orElse(0);
        System.out.printf(Locale.ROOT, "  ladder %d: cold %.1f  warm %s%n", L, rate[0],
            Arrays.toString(Arrays.copyOfRange(rate, 1, rate.length)));
      }
      Arrays.sort(cold); Arrays.sort(warm);
      double c = cold[ladders / 2], w = warm[ladders / 2];
      System.out.printf(Locale.ROOT,
          "RESULT engine=%s product=%s steps=%d paths=%,d pass=%s crn=%s | cold %.2f Mpath/s | warm %.2f Mpath/s | warm/cold %.2fx | price@100 %.17g delta %.17g%n",
          engine, product, steps, paths, greeks ? "value+5G" : "price",
          System.getProperty("nablatensor.crn", "off"), c, w, w / c, lastPrice, lastDelta);
    }
  }
}
