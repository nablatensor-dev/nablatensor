import com.nablatensor.quant.*;
import java.util.Locale;
/** Full Greek set at spot 100, seed 42, for the agreement table. */
public final class Greeks {
  public static void main(String[] a) {
    EquityMarket base = EquityMarket.atmOneYear();
    for (String[] p : new String[][] {{"european", "1", "20000000"}, {"asian", "252", "300000"}}) {
      var prod = p[0].equals("european") ? Products.europeanCall() : Products.asianCall();
      try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(prod).market(base).steps(Integer.parseInt(p[1])).fp64().threads(8).greeks().on("cpu-jit").build()) {
        var r = mc.run(base, Long.parseLong(p[2]), 42L);
        System.out.printf(Locale.ROOT, "%-9s paths=%,d price %.6f delta %.6f dV/dK %.6f vega %.6f rho %.6f dV/dT %.6f  stderr %.5f%n",
            p[0], Long.parseLong(p[2]), r.price(), r.greek(EquityMarket::spot), r.greek(EquityMarket::strike),
            r.greek(EquityMarket::vol), r.greek(EquityMarket::rate), r.greek(EquityMarket::maturity), r.standardError());
      }
    }
  }
}
