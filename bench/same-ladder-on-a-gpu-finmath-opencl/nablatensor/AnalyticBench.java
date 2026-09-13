import java.util.Locale;
import org.jquantlib.Settings;
import org.jquantlib.daycounters.Actual365Fixed;
import org.jquantlib.daycounters.DayCounter;
import org.jquantlib.exercise.EuropeanExercise;
import org.jquantlib.instruments.EuropeanOption;
import org.jquantlib.instruments.Option;
import org.jquantlib.instruments.Payoff;
import org.jquantlib.instruments.PlainVanillaPayoff;
import org.jquantlib.pricingengines.AnalyticEuropeanEngine;
import org.jquantlib.processes.BlackScholesMertonProcess;
import org.jquantlib.quotes.Handle;
import org.jquantlib.quotes.Quote;
import org.jquantlib.quotes.SimpleQuote;
import org.jquantlib.termstructures.BlackVolTermStructure;
import org.jquantlib.termstructures.YieldTermStructure;
import org.jquantlib.termstructures.volatilities.BlackConstantVol;
import org.jquantlib.termstructures.yieldcurves.FlatForward;
import org.jquantlib.time.Date;
import org.jquantlib.time.Month;
import org.jquantlib.time.calendars.NullCalendar;
import com.opengamma.strata.pricer.impl.option.BlackScholesFormulaRepository;

/** Closed-form European call, price + Greeks, one thread, as shipped: JQuantLib AnalyticEuropeanEngine and Strata BlackScholesFormulaRepository. */
public final class AnalyticBench {
  static final double[] SPOTS = {98, 99, 99.5, 100, 100.5, 101, 102};

  public static void main(String[] a) {
    String lib = System.getProperty("lib", "strata");
    int reps = Integer.getInteger("reps", 2_000_000);   // valuations per timing
    if ("strata".equals(lib)) strata(reps); else jquantlib(reps);
  }

  static void strata(int reps) {
    double sink = 0;
    for (int w = 0; w < 3; w++) sink += strataLadder(reps);
    double best = Double.MAX_VALUE;
    for (int r = 0; r < 5; r++) {
      long t0 = System.nanoTime();
      sink += strataLadder(reps);
      best = Math.min(best, (System.nanoTime() - t0) / 1e9);
    }
    double p = BlackScholesFormulaRepository.price(100, 100, 1.0, 0.20, 0.03, 0.03, true);
    double d = BlackScholesFormulaRepository.delta(100, 100, 1.0, 0.20, 0.03, 0.03, true);
    double v = BlackScholesFormulaRepository.vega(100, 100, 1.0, 0.20, 0.03, 0.03);
    double rho = BlackScholesFormulaRepository.rho(100, 100, 1.0, 0.20, 0.03, 0.03, true);
    double th = BlackScholesFormulaRepository.theta(100, 100, 1.0, 0.20, 0.03, 0.03, true);
    double dK = BlackScholesFormulaRepository.dualDelta(100, 100, 1.0, 0.20, 0.03, 0.03, true);
    System.out.printf(Locale.ROOT, "RESULT lib=strata closed-form price+5G one thread | %.2f M valuations/s | price %.10f delta %.10f dK %.6f vega %.6f rho %.6f theta %.6f (sink %.1f)%n",
        (double) reps / best / 1e6, p, d, dK, v, rho, th, sink);
  }

  static double strataLadder(int reps) {
    double s = 0;
    for (int i = 0; i < reps; i++) {
      double spot = SPOTS[i % SPOTS.length];
      s += BlackScholesFormulaRepository.price(spot, 100, 1.0, 0.20, 0.03, 0.03, true)
         + BlackScholesFormulaRepository.delta(spot, 100, 1.0, 0.20, 0.03, 0.03, true)
         + BlackScholesFormulaRepository.dualDelta(spot, 100, 1.0, 0.20, 0.03, 0.03, true)
         + BlackScholesFormulaRepository.vega(spot, 100, 1.0, 0.20, 0.03, 0.03)
         + BlackScholesFormulaRepository.rho(spot, 100, 1.0, 0.20, 0.03, 0.03, true)
         + BlackScholesFormulaRepository.theta(spot, 100, 1.0, 0.20, 0.03, 0.03, true);
    }
    return s;
  }

  static void jquantlib(int reps) {
    Date today = new Date(11, Month.September, 2026);
    new Settings().setEvaluationDate(today);
    Date maturity = new Date(11, Month.September, 2027);
    DayCounter dc = new Actual365Fixed();
    SimpleQuote spotQ = new SimpleQuote(100.0);
    Handle<Quote> spot = new Handle<Quote>(spotQ);
    Handle<YieldTermStructure> q = new Handle<YieldTermStructure>(new FlatForward(today, 0.0, dc));
    Handle<YieldTermStructure> r = new Handle<YieldTermStructure>(new FlatForward(today, 0.03, dc));
    Handle<BlackVolTermStructure> vol = new Handle<BlackVolTermStructure>(new BlackConstantVol(today, new NullCalendar(), 0.20, dc));
    BlackScholesMertonProcess process = new BlackScholesMertonProcess(spot, q, r, vol);
    Payoff payoff = new PlainVanillaPayoff(Option.Type.Call, 100.0);
    EuropeanOption option = new EuropeanOption(payoff, new EuropeanExercise(maturity));
    option.setPricingEngine(new AnalyticEuropeanEngine(process));

    double sink = 0;
    for (int w = 0; w < 3; w++) sink += jqLadder(option, spotQ, reps);
    double best = Double.MAX_VALUE;
    for (int rr = 0; rr < 5; rr++) {
      long t0 = System.nanoTime();
      sink += jqLadder(option, spotQ, reps);
      best = Math.min(best, (System.nanoTime() - t0) / 1e9);
    }
    spotQ.setValue(100.0);
    System.out.printf(Locale.ROOT, "RESULT lib=jquantlib closed-form price+Greeks one thread | %.3f M valuations/s | price %.10f delta %.10f dK %s vega %.6f rho %.6f theta %.6f (sink %.1f)%n",
        (double) reps / best / 1e6, option.NPV(), option.delta(), "n/a", option.vega(), option.rho(), option.theta(), sink);
  }

  static double jqLadder(EuropeanOption option, SimpleQuote spotQ, int reps) {
    double s = 0;
    for (int i = 0; i < reps; i++) {
      spotQ.setValue(SPOTS[i % SPOTS.length]);     // notifies observers; the option recalculates lazily
      s += option.NPV() + option.delta() + option.vega() + option.rho() + option.theta();
    }
    return s;
  }
}
