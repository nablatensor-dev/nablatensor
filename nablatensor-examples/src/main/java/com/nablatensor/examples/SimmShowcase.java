/*
 * Copyright 2026 The NablaTensor Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nablatensor.examples;

import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Products;
import com.nablatensor.risk.NestedAggregation;
import com.nablatensor.risk.RiskClassEnum;
import com.nablatensor.risk.RiskFactor;
import com.nablatensor.risk.RiskMeasureEnum;
import com.nablatensor.risk.Sensitivities;
import com.nablatensor.scenario.Scenario;
import com.nablatensor.scenario.ScenarioRunner;
import com.nablatensor.scenario.ScenarioSet;
import com.nablatensor.scenario.Shock;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * A compact, end-to-end ISDA SIMM demonstration built around the one expensive
 * stage of an initial-margin run: generating the full delta / vega / curvature
 * sensitivity set for the whole non-cleared book, and its re-run once per risk
 * factor when the sensitivities are taken by prescribed bump instead of one
 * adjoint reverse sweep.
 *
 * <p>The narrated {@code demo/isda-simm-full.sh} walks twelve stages on
 * whichever adjoint backend is fastest here; this class is the non-narrated
 * runner and the data provider it shares. Every number is produced on the
 * machine you run it on.
 *
 * <p>SIMM's six risk classes are mapped onto the shared FRTB / SIMM
 * {@link RiskClassEnum} enum: {@code GIRR} = Interest Rate, {@code FX} = FX,
 * {@code CSR_NON_SEC} = Credit Qualifying, {@code CSR_SEC} = Credit
 * Non-Qualifying, {@code EQUITY} = Equity, {@code COMMODITY} = Commodity.
 *
 * <p><b>Calculators, not sign-off.</b> Parameter tables are indicative demo
 * values, not the ISDA-published calibration; model validation, parameter
 * attestation and margin dispute resolution stay with the user.
 *
 * <p>Run: {@code mvn -q -o -pl nablatensor-examples exec:java
 * -Dexec.mainClass=com.nablatensor.examples.SimmShowcase}. System properties:
 * {@code -Dengine=}, {@code -Dpaths=}, {@code -Dsteps=}, {@code -Dseed=},
 * {@code -Dbacktest.days=}.
 */
public final class SimmShowcase {

  private static final String PARAMETER_VERSION = "DEMO-SIMM-v2.6-2026-09";

  /** The SIMM risk classes, in report order, on the shared enum. */
  public static final List<RiskClassEnum> SIMM_CLASSES = List.of(
      RiskClassEnum.GIRR, RiskClassEnum.CSR_NON_SEC, RiskClassEnum.CSR_SEC,
      RiskClassEnum.EQUITY, RiskClassEnum.COMMODITY, RiskClassEnum.FX);

  private static final String BOOK_CSV = """
      id,counterparty,product,notional,side
      IRS-USD-10Y,CPTY-ALPHA,RATES_FX,120000000,1
      XCCY-EURUSD-5Y,CPTY-ALPHA,RATES_FX,80000000,-1
      CDS-IG-CPTY-5Y,CPTY-BRAVO,CREDIT,50000000,1
      CDX-HY-INDEX-5Y,CPTY-BRAVO,CREDIT,25000000,-1
      EQ-CALL-SX5E-1Y,CPTY-CHARLIE,EQUITY,15000000,-1
      EQ-PUT-SPX-1Y,CPTY-CHARLIE,EQUITY,10000000,1
      COMDTY-WTI-SWAP-2Y,CPTY-DELTA,COMMODITY,30000000,1
      """;

  private SimmShowcase() {
  }

  // ---- shared demo inputs -----------------------------------------------

  public static SimmParameters parameters() {
    return SimmParameters.demoV26();
  }

  public static List<TradeSpec> book() {
    return TradeSpec.load(BOOK_CSV);
  }

  /** The four SIMM product classes and the risk classes each rolls up. */
  public enum ProductClassEnum {
    /** Interest-rate and FX products; rolls up GIRR and FX. */
    RATES_FX(RiskClassEnum.GIRR, RiskClassEnum.FX),
    /** Credit products; rolls up non-securitisation and securitisation CSR. */
    CREDIT(RiskClassEnum.CSR_NON_SEC, RiskClassEnum.CSR_SEC),
    /** Equity products; rolls up the equity risk class. */
    EQUITY(RiskClassEnum.EQUITY),
    /** Commodity products; rolls up the commodity risk class. */
    COMMODITY(RiskClassEnum.COMMODITY);

    private final List<RiskClassEnum> riskClasses;

    ProductClassEnum(RiskClassEnum... riskClasses) {
      this.riskClasses = List.of(riskClasses);
    }

    public List<RiskClassEnum> riskClasses() {
      return riskClasses;
    }
  }

  /** Indicative SIMM risk-class correlation for the within-product pairs. */
  public static double psi(RiskClassEnum a, RiskClassEnum b) {
    if (a == b) {
      return 1.0;
    }
    var pair = java.util.EnumSet.of(a, b);
    if (pair.equals(java.util.EnumSet.of(RiskClassEnum.GIRR, RiskClassEnum.FX))) {
      return 0.28;
    }
    if (pair.equals(java.util.EnumSet.of(RiskClassEnum.CSR_NON_SEC, RiskClassEnum.CSR_SEC))) {
      return 0.15;
    }
    return 0.0;
  }

  // ---- the one expensive stage ----------------------------------------

  /**
   * The book sensitivity two ways. One adjoint reverse sweep <b>per trade</b>
   * over a {@code bookTrades}-trade netting set produces the whole Greek vector;
   * then the same book is revalued once per prescribed risk-factor bump. The
   * sweep time and the bump time are the demo's headline comparison, and they
   * are sized to a bank-scale daily SIMM run rather than a toy.
   *
   * <p>One compiled kernel is replayed across the book — {@code bookTrades} x
   * {@code pathsPerTrade} path valuations for the sweep, and
   * {@code bookTrades} x (bumps) for the letter-compliant alternative.
   */
  public static HeavyRun runHeavySensitivities(String engine, boolean fp32, int bookTrades,
                                               long pathsPerTrade, int steps, long seed) {
    EquityMarket base = EquityMarket.atmOneYear();
    var greekBuild = MonteCarlo.of(Products.asianCall()).market(base).steps(steps);
    var priceBuild = MonteCarlo.of(Products.asianCall()).market(base).steps(steps);
    MonteCarlo<EquityMarket> greeks = fp32
        ? greekBuild.fp32().greeks().on(engine).build()
        : greekBuild.fp64().greeks().on(engine).build();
    MonteCarlo<EquityMarket> pricer = fp32
        ? priceBuild.fp32().priceOnly().on(engine).build()
        : priceBuild.fp64().priceOnly().on(engine).build();

    List<Scenario> bumps = new ArrayList<>();
    bumps.add(Scenario.of("base"));
    for (String input : new String[] {"spot", "vol", "rate", "strike", "maturity"}) {
      bumps.add(Scenario.of(input + "+", Shock.relative(input, 1.0e-4)));
      bumps.add(Scenario.of(input + "-", Shock.relative(input, -1.0e-4)));
    }
    int prescribedBumps = bumps.size() - 1;
    bumps.add(Scenario.of("spotUp1pct", Shock.relative("spot", 0.01)));
    bumps.add(Scenario.of("spotDn1pct", Shock.relative("spot", -0.01)));
    ScenarioSet shocks = ScenarioSet.list(bumps.toArray(new Scenario[0]));

    try (greeks; pricer) {
      long warmup = Math.min(pathsPerTrade, 20_000L);
      greeks.run(warmup, seed);
      pricer.run(warmup, seed);

      double delta = 0.0;
      double vega = 0.0;
      double irDelta = 0.0;
      long sweepStart = System.nanoTime();
      for (int trade = 0; trade < bookTrades; trade++) {
        Nabla.TypedValuation<EquityMarket> risk =
            greeks.run(tradeMarket(base, trade), pathsPerTrade, seed + trade);
        delta += risk.greek(EquityMarket::spot);
        vega += risk.greek(EquityMarket::vol) * 0.01;
        irDelta += risk.greek(EquityMarket::rate) * 0.01;
      }
      double sweepSeconds = (System.nanoTime() - sweepStart) / 1.0e9;

      double curvature = 0.0;
      double bumpDelta = 0.0;
      double bumpVega = 0.0;
      long bumpStart = System.nanoTime();
      for (int trade = 0; trade < bookTrades; trade++) {
        EquityMarket market = tradeMarket(base, trade);
        Map<String, Nabla.TypedValuation<EquityMarket>> pv =
            ScenarioRunner.run(pricer, market, shocks, pathsPerTrade, seed + trade);
        double basePrice = pv.get("base").price();
        curvature += pv.get("spotUp1pct").price() + pv.get("spotDn1pct").price() - 2.0 * basePrice;
        bumpDelta += (pv.get("spot+").price() - pv.get("spot-").price())
            / (2.0 * market.spot() * 1.0e-4);
        bumpVega += (pv.get("vol+").price() - pv.get("vol-").price())
            / (2.0 * market.vol() * 1.0e-4) * 0.01;
      }
      double bumpSeconds = (System.nanoTime() - bumpStart) / 1.0e9;

      return HeavyRun.of().engine(engine).bookTrades(bookTrades).pathsPerTrade(pathsPerTrade).steps(steps).delta(delta).vega(vega).irDelta(irDelta).curvature(curvature).bumpDelta(bumpDelta).bumpVega(bumpVega).sweepSeconds(sweepSeconds).bumpSeconds(bumpSeconds).revaluations(bookTrades * prescribedBumps).build();
    }
  }

  /** A slightly different equity trade per book slot — spread of spot, vol, rate, maturity. */
  private static EquityMarket tradeMarket(EquityMarket base, int trade) {
    return EquityMarket.of().spot(base.spot() * (0.85 + 0.012 * (trade % 25))).strike(base.strike()).vol(base.vol() * (0.8 + 0.05 * (trade % 7))).rate(base.rate() + 0.0015 * (trade % 6)).maturity(base.maturity() * (0.5 + 0.15 * (trade % 10))).build();
  }

  // ---- CRIF: trades -> the sensitivity vector -------------------------

  /**
   * A demo CRIF (Common Risk Interchange Format) sensitivity vector. Synthetic
   * per-bucket delta / vega / curvature entries scaled by trade notional and
   * side, with the machine-produced Greeks from {@link #runHeavySensitivities}
   * overlaid onto the equity and rates factors so at least one factor per
   * measure carries a real adjoint number.
   */
  public static Sensitivities buildSensitivities(List<TradeSpec> specs, SimmParameters params,
                                                 HeavyRun heavy) {
    Sensitivities.Builder crif = Sensitivities.builder();
    for (int tradeIndex = 0; tradeIndex < specs.size(); tradeIndex++) {
      TradeSpec spec = specs.get(tradeIndex);
      double scale = spec.side() * spec.notional() / 1_000_000.0;
      int factorIndex = 0;
      for (RiskClassEnum riskClass : ProductClassEnum.valueOf(spec.product()).riskClasses()) {
        Map<String, BucketParameter> table = params.tables().get(riskClass);
        List<String> buckets = new ArrayList<>(table.keySet());
        for (int b = 0; b < Math.min(2, buckets.size()); b++) {
          String bucket = buckets.get(b);
          double alternating = ((factorIndex++ + tradeIndex) & 1) == 0 ? 1.0 : -0.6;
          String name = spec.id() + "#" + bucket;
          crif.add(RiskFactor.of().riskClass(riskClass).measure(RiskMeasureEnum.DELTA).bucket(bucket).name(name).tenor(0.0).tenor2(0.0).build(),
              scale * alternating * 0.02);
          crif.add(RiskFactor.of().riskClass(riskClass).measure(RiskMeasureEnum.VEGA).bucket(bucket).name(name).tenor(1.0).tenor2(0.0).build(),
              Math.abs(scale) * 0.004 * (1.0 + 0.1 * tradeIndex));
          crif.add(RiskFactor.of().riskClass(riskClass).measure(RiskMeasureEnum.CURVATURE).bucket(bucket).name(name).tenor(0.0).tenor2(0.0).build(),
              Math.abs(scale) * alternating * 0.0015);
        }
      }
    }
    crif.add(RiskFactor.equityDelta("4", "EQ-CALL-SX5E-1Y"), heavy.delta());
    crif.add(RiskFactor.of().riskClass(RiskClassEnum.EQUITY).measure(RiskMeasureEnum.VEGA).bucket("4").name("EQ-CALL-SX5E-1Y").tenor(1.0).tenor2(0.0).build(), heavy.vega());
    crif.add(RiskFactor.equityDelta("4", "EQ-CALL-SX5E-1Y").asCurvature(), heavy.curvature());
    crif.add(RiskFactor.girrDelta("USD", 10.0), heavy.irDelta());
    return crif.build();
  }

  // ---- SIMM aggregation ---------------------------------------------

  public static SimmResult simm(SimmParameters params, Sensitivities crif) {
    EnumMap<RiskClassEnum, EnumMap<RiskMeasureEnum, Double>> byClass = new EnumMap<>(RiskClassEnum.class);
    for (RiskClassEnum riskClass : SIMM_CLASSES) {
      Map<String, BucketParameter> table = params.tables().get(riskClass);
      Sensitivities classRisk = crif.ofClass(riskClass);
      EnumMap<RiskMeasureEnum, Double> byMeasure = new EnumMap<>(RiskMeasureEnum.class);
      for (RiskMeasureEnum measure : RiskMeasureEnum.values()) {
        Sensitivities cm = classRisk.ofMeasure(measure);
        byMeasure.put(measure, marginFor(table, cm, measure));
      }
      byClass.put(riskClass, byMeasure);
    }

    EnumMap<ProductClassEnum, EnumMap<RiskMeasureEnum, Double>> byProduct = new EnumMap<>(ProductClassEnum.class);
    EnumMap<ProductClassEnum, Double> productTotal = new EnumMap<>(ProductClassEnum.class);
    double simm = 0.0;
    for (ProductClassEnum product : ProductClassEnum.values()) {
      EnumMap<RiskMeasureEnum, Double> im = new EnumMap<>(RiskMeasureEnum.class);
      double total = 0.0;
      for (RiskMeasureEnum measure : RiskMeasureEnum.values()) {
        double sum = 0.0;
        for (RiskClassEnum r : product.riskClasses()) {
          double kr = byClass.get(r).get(measure);
          sum += kr * kr;
          for (RiskClassEnum s : product.riskClasses()) {
            if (r != s) {
              sum += psi(r, s) * kr * byClass.get(s).get(measure);
            }
          }
        }
        double value = Math.sqrt(Math.max(0.0, sum));
        im.put(measure, value);
        total += value;
      }
      byProduct.put(product, im);
      productTotal.put(product, total);
      simm += total;
    }
    return new SimmResult(byClass, byProduct, productTotal, simm);
  }

  private static double marginFor(Map<String, BucketParameter> table, Sensitivities cm,
                                  RiskMeasureEnum measure) {
    double gamma = table.values().iterator().next().gamma();
    if (measure == RiskMeasureEnum.CURVATURE) {
      return NestedAggregation.curvature(
          (k, l) -> k.equals(l) ? 1.0 : table.get(k.bucket()).rho(),
          (b, c) -> b.equals(c) ? 1.0 : gamma).aggregate(cm).total();
    }
    double threshold = measure == RiskMeasureEnum.DELTA
        ? table.values().iterator().next().deltaThreshold()
        : table.values().iterator().next().vegaThreshold();
    NestedAggregation aggregation = NestedAggregation.delta(
        factor -> measure == RiskMeasureEnum.DELTA
            ? table.get(factor.bucket()).deltaRw()
            : table.get(factor.bucket()).vegaRw(),
        (k, l) -> k.equals(l) ? 1.0 : table.get(k.bucket()).rho(),
        (b, c) -> b.equals(c) ? 1.0 : gamma)
        .withConcentration(concentrationFor(cm, threshold));
    return aggregation.aggregate(cm).total();
  }

  /** SIMM concentration risk factor {@code CR_b = max(1, sqrt(|sum s_b| / T_b))}. */
  private static NestedAggregation.ConcentrationFactor concentrationFor(Sensitivities cm,
                                                                       double threshold) {
    Map<String, Double> bucketSum = new LinkedHashMap<>();
    cm.asMap().forEach((factor, sensitivity) ->
        bucketSum.merge(factor.bucket(), sensitivity, Double::sum));
    Map<String, Double> cr = new LinkedHashMap<>();
    bucketSum.forEach((bucket, sum) ->
        cr.put(bucket, Math.max(1.0, Math.sqrt(Math.abs(sum) / threshold))));
    return (factor, sensitivity) -> cr.getOrDefault(factor.bucket(), 1.0);
  }

  // ---- backtest: the SIMM calculation replayed over history ----------

  /**
   * The SIMM calculation replayed over a run of pseudo-historical sensitivity
   * snapshots — the shape of the ISDA annual backtest, which recomputes SIMM IM
   * and clean P&L over years of daily history for every counterparty.
   */
  public static BacktestResult backtest(SimmParameters params, Sensitivities crif,
                                        int days, long seed) {
    Random random = new Random(seed);
    double im0 = simm(params, crif).total();
    long start = System.nanoTime();
    int breaches = 0;
    for (int day = 0; day < days; day++) {
      double move = 0.05 * random.nextGaussian();
      double im = simm(params, crif.scaled(1.0 + move)).total();
      double cleanPnl = (move + 0.09 * random.nextGaussian()) * im0 * 4.5;
      if (Math.abs(cleanPnl) > im) {
        breaches++;
      }
    }
    double seconds = (System.nanoTime() - start) / 1.0e9;
    return new BacktestResult(days, breaches, seconds, im0);
  }

  // ---- non-narrated runner -----------------------------------------

  public static void main(String[] args) {
    String engine = System.getProperty("engine", "cpu-jit");
    boolean gpu = !engine.equals("cpu-jit");
    int bookTrades = Integer.getInteger("book.trades", gpu ? 24 : 6);
    long paths = Long.getLong("paths", gpu ? 1_300_000L : 400_000L);
    int steps = Integer.getInteger("steps", 252);
    long seed = Long.getLong("seed", 42L);
    int backtestDays = Integer.getInteger("backtest.days", 750);

    SimmParameters params = parameters();
    List<TradeSpec> book = book();

    System.out.printf(Locale.ROOT,
        "ISDA SIMM  |  %s  |  engine %s  |  %d-trade book x %,d paths x %d steps%n%n",
        PARAMETER_VERSION, engine, bookTrades, paths, steps);

    System.out.println("[1/6] THE NON-CLEARED BOOK");
    for (TradeSpec trade : book) {
      System.out.printf(Locale.ROOT, "      %-20s %-13s %-9s side=%+2.0f  $%.0fm%n",
          trade.id(), trade.counterparty(), trade.product(), trade.side(),
          trade.notional() / 1_000_000.0);
    }

    System.out.printf(Locale.ROOT, "%n[2/6] GENERATE THE FULL SENSITIVITY SET ON %s%n",
        engine.toUpperCase(Locale.ROOT));
    HeavyRun heavy = runHeavySensitivities(engine, true, bookTrades, paths, steps, seed);
    System.out.printf(Locale.ROOT,
        "      one adjoint sweep per trade   %.3f s  -> delta+vega+rho+curvature (%,d path valuations)%n",
        heavy.sweepSeconds(), heavy.pathValuations());
    System.out.printf(Locale.ROOT, "      %d prescribed bump revaluations %.3f s%n",
        heavy.revaluations(), heavy.bumpSeconds());

    Sensitivities crif = buildSensitivities(book, params, heavy);
    System.out.printf(Locale.ROOT, "%n[3/6] CRIF: %d risk-factor sensitivities across %d SIMM classes%n",
        crif.asMap().size(), SIMM_CLASSES.size());

    SimmResult result = simm(params, crif);
    System.out.println("\n[4/6] MARGIN BY RISK CLASS  (delta / vega / curvature)");
    for (RiskClassEnum riskClass : SIMM_CLASSES) {
      EnumMap<RiskMeasureEnum, Double> k = result.byClass().get(riskClass);
      System.out.printf(Locale.ROOT, "      %-12s %10.3f %10.3f %10.3f%n", riskClass,
          k.get(RiskMeasureEnum.DELTA), k.get(RiskMeasureEnum.VEGA), k.get(RiskMeasureEnum.CURVATURE));
    }

    System.out.println("\n[5/6] IM BY PRODUCT CLASS, THEN TOTAL SIMM ($m)");
    for (ProductClassEnum product : ProductClassEnum.values()) {
      System.out.printf(Locale.ROOT, "      %-10s %10.3f%n", product,
          result.productTotal().get(product));
    }
    System.out.printf(Locale.ROOT, "      %-10s %10.3f%n", "SIMM", result.total());

    BacktestResult bt = backtest(params, crif, backtestDays, seed);
    System.out.printf(Locale.ROOT,
        "%n[6/6] BACKTEST: %d historical snapshots in %.3f s   (%d P&L breaches)%n",
        bt.days(), bt.seconds(), bt.breaches());
    System.out.println("      Indicative tables; not the ISDA-published SIMM calibration.");
  }

  // ---- demo-local records ----------------------------------------

  public static final class HeavyRun {
    private final String engine;
    private final int bookTrades;
    private final long pathsPerTrade;
    private final int steps;
    private final double delta;
    private final double vega;
    private final double irDelta;
    private final double curvature;
    private final double bumpDelta;
    private final double bumpVega;
    private final double sweepSeconds;
    private final double bumpSeconds;
    private final int revaluations;

    private HeavyRun(String engine, int bookTrades, long pathsPerTrade, int steps, double delta, double vega, double irDelta, double curvature, double bumpDelta, double bumpVega, double sweepSeconds, double bumpSeconds, int revaluations) {
      this.engine = engine;
      this.bookTrades = bookTrades;
      this.pathsPerTrade = pathsPerTrade;
      this.steps = steps;
      this.delta = delta;
      this.vega = vega;
      this.irDelta = irDelta;
      this.curvature = curvature;
      this.bumpDelta = bumpDelta;
      this.bumpVega = bumpVega;
      this.sweepSeconds = sweepSeconds;
      this.bumpSeconds = bumpSeconds;
      this.revaluations = revaluations;
    }

    public static Builder of() { return new Builder(); }

    public String engine() { return engine; }

    public int bookTrades() { return bookTrades; }

    public long pathsPerTrade() { return pathsPerTrade; }

    public int steps() { return steps; }

    public double delta() { return delta; }

    public double vega() { return vega; }

    public double irDelta() { return irDelta; }

    public double curvature() { return curvature; }

    public double bumpDelta() { return bumpDelta; }

    public double bumpVega() { return bumpVega; }

    public double sweepSeconds() { return sweepSeconds; }

    public double bumpSeconds() { return bumpSeconds; }

    public int revaluations() { return revaluations; }

    public long pathValuations() {
      return (long) bookTrades * pathsPerTrade;
    }

    public static final class Builder {
      private String engine;
      private boolean engineSet;
      private int bookTrades;
      private boolean bookTradesSet;
      private long pathsPerTrade;
      private boolean pathsPerTradeSet;
      private int steps;
      private boolean stepsSet;
      private double delta;
      private boolean deltaSet;
      private double vega;
      private boolean vegaSet;
      private double irDelta;
      private boolean irDeltaSet;
      private double curvature;
      private boolean curvatureSet;
      private double bumpDelta;
      private boolean bumpDeltaSet;
      private double bumpVega;
      private boolean bumpVegaSet;
      private double sweepSeconds;
      private boolean sweepSecondsSet;
      private double bumpSeconds;
      private boolean bumpSecondsSet;
      private int revaluations;
      private boolean revaluationsSet;

      public Builder engine(String value) {
        this.engine = value;
        this.engineSet = true;
        return this;
      }

      public Builder bookTrades(int value) {
        this.bookTrades = value;
        this.bookTradesSet = true;
        return this;
      }

      public Builder pathsPerTrade(long value) {
        this.pathsPerTrade = value;
        this.pathsPerTradeSet = true;
        return this;
      }

      public Builder steps(int value) {
        this.steps = value;
        this.stepsSet = true;
        return this;
      }

      public Builder delta(double value) {
        this.delta = value;
        this.deltaSet = true;
        return this;
      }

      public Builder vega(double value) {
        this.vega = value;
        this.vegaSet = true;
        return this;
      }

      public Builder irDelta(double value) {
        this.irDelta = value;
        this.irDeltaSet = true;
        return this;
      }

      public Builder curvature(double value) {
        this.curvature = value;
        this.curvatureSet = true;
        return this;
      }

      public Builder bumpDelta(double value) {
        this.bumpDelta = value;
        this.bumpDeltaSet = true;
        return this;
      }

      public Builder bumpVega(double value) {
        this.bumpVega = value;
        this.bumpVegaSet = true;
        return this;
      }

      public Builder sweepSeconds(double value) {
        this.sweepSeconds = value;
        this.sweepSecondsSet = true;
        return this;
      }

      public Builder bumpSeconds(double value) {
        this.bumpSeconds = value;
        this.bumpSecondsSet = true;
        return this;
      }

      public Builder revaluations(int value) {
        this.revaluations = value;
        this.revaluationsSet = true;
        return this;
      }

      public Builder from(HeavyRun value) {
        if (value == null) throw new NullPointerException("value");
        engine(value.engine());
        bookTrades(value.bookTrades());
        pathsPerTrade(value.pathsPerTrade());
        steps(value.steps());
        delta(value.delta());
        vega(value.vega());
        irDelta(value.irDelta());
        curvature(value.curvature());
        bumpDelta(value.bumpDelta());
        bumpVega(value.bumpVega());
        sweepSeconds(value.sweepSeconds());
        bumpSeconds(value.bumpSeconds());
        revaluations(value.revaluations());
        return this;
      }

      public HeavyRun build() {
        if (!engineSet) throw new IllegalStateException("Missing required value: engine");
        if (!bookTradesSet) throw new IllegalStateException("Missing required value: bookTrades");
        if (!pathsPerTradeSet) throw new IllegalStateException("Missing required value: pathsPerTrade");
        if (!stepsSet) throw new IllegalStateException("Missing required value: steps");
        if (!deltaSet) throw new IllegalStateException("Missing required value: delta");
        if (!vegaSet) throw new IllegalStateException("Missing required value: vega");
        if (!irDeltaSet) throw new IllegalStateException("Missing required value: irDelta");
        if (!curvatureSet) throw new IllegalStateException("Missing required value: curvature");
        if (!bumpDeltaSet) throw new IllegalStateException("Missing required value: bumpDelta");
        if (!bumpVegaSet) throw new IllegalStateException("Missing required value: bumpVega");
        if (!sweepSecondsSet) throw new IllegalStateException("Missing required value: sweepSeconds");
        if (!bumpSecondsSet) throw new IllegalStateException("Missing required value: bumpSeconds");
        if (!revaluationsSet) throw new IllegalStateException("Missing required value: revaluations");
        return new HeavyRun(engine, bookTrades, pathsPerTrade, steps, delta, vega, irDelta, curvature, bumpDelta, bumpVega, sweepSeconds, bumpSeconds, revaluations);
      }
    }
  }

  public record BacktestResult(int days, int breaches, double seconds, double im0) {
  }

  public record SimmResult(Map<RiskClassEnum, EnumMap<RiskMeasureEnum, Double>> byClass,
                           Map<ProductClassEnum, EnumMap<RiskMeasureEnum, Double>> byProduct,
                           Map<ProductClassEnum, Double> productTotal,
                           double total) {
  }

  public record BucketParameter(double deltaRw, double vegaRw, double rho, double gamma,
                                double deltaThreshold, double vegaThreshold) {
  }

  public record SimmParameters(String version, Map<RiskClassEnum, Map<String, BucketParameter>> tables) {
    public SimmParameters {
      EnumMap<RiskClassEnum, Map<String, BucketParameter>> copy = new EnumMap<>(RiskClassEnum.class);
      tables.forEach((riskClass, table) ->
          copy.put(riskClass, Collections.unmodifiableMap(new LinkedHashMap<>(table))));
      tables = Collections.unmodifiableMap(copy);
    }

    public int bucketCount() {
      return tables.values().stream().mapToInt(Map::size).sum();
    }

    public static SimmParameters demoV26() {
      EnumMap<RiskClassEnum, Map<String, BucketParameter>> tables = new EnumMap<>(RiskClassEnum.class);
      tables.put(RiskClassEnum.GIRR, named(new String[] {"USD", "EUR", "JPY"},
          new double[] {0.011, 0.011, 0.011}, 0.18, 0.63, 0.24, 330.0, 130.0));
      tables.put(RiskClassEnum.CSR_NON_SEC, numbered(
          new double[] {0.008, 0.012, 0.017, 0.024, 0.033, 0.045, 0.055},
          0.64, 0.42, 0.21, 12.0, 320.0));
      tables.put(RiskClassEnum.CSR_SEC, numbered(
          new double[] {0.28, 0.66}, 0.64, 0.27, 0.35, 3.0, 85.0));
      tables.put(RiskClassEnum.EQUITY, numbered(
          new double[] {0.22, 0.28, 0.34, 0.40}, 0.28, 0.16, 0.15, 8.0, 210.0));
      tables.put(RiskClassEnum.COMMODITY, numbered(
          new double[] {0.19, 0.30, 0.44, 0.60, 0.83}, 0.42, 0.31, 0.23, 260.0, 74.0));
      tables.put(RiskClassEnum.FX, named(new String[] {"USD", "EUR", "JPY"},
          new double[] {0.075, 0.075, 0.075}, 0.30, 0.50, 0.50, 2400.0, 480.0));
      return new SimmParameters(PARAMETER_VERSION, tables);
    }

    private static Map<String, BucketParameter> numbered(double[] weights, double vega,
                                                         double rho, double gamma,
                                                         double deltaThreshold,
                                                         double vegaThreshold) {
      String[] names = new String[weights.length];
      for (int index = 0; index < names.length; index++) {
        names[index] = Integer.toString(index + 1);
      }
      return named(names, weights, vega, rho, gamma, deltaThreshold, vegaThreshold);
    }

    private static Map<String, BucketParameter> named(String[] names, double[] weights,
                                                      double vega, double rho, double gamma,
                                                      double deltaThreshold,
                                                      double vegaThreshold) {
      Map<String, BucketParameter> table = new LinkedHashMap<>();
      for (int index = 0; index < names.length; index++) {
        table.put(names[index],
            new BucketParameter(weights[index], vega, rho, gamma, deltaThreshold, vegaThreshold));
      }
      return table;
    }
  }

  public record TradeSpec(String id, String counterparty, String product,
                          double notional, double side) {
    public static List<TradeSpec> load(String csv) {
      try (BufferedReader reader = new BufferedReader(new StringReader(csv))) {
        return reader.lines().skip(1).filter(line -> !line.isBlank())
            .map(line -> line.split(",", -1))
            .map(row -> new TradeSpec(row[0], row[1], row[2],
                Double.parseDouble(row[3]), Double.parseDouble(row[4])))
            .toList();
      } catch (IOException exception) {
        throw new IllegalStateException("cannot load embedded demo CSV", exception);
      }
    }
  }
}
