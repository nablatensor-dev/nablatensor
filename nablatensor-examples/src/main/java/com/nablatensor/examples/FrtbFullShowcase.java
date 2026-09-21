package com.nablatensor.examples;

import com.nablatensor.engine.Nabla;
import com.nablatensor.quant.EquityMarket;
import com.nablatensor.quant.MonteCarlo;
import com.nablatensor.quant.Products;
import com.nablatensor.risk.CorrelationScenarioEnum;
import com.nablatensor.risk.NestedAggregation;
import com.nablatensor.risk.Portfolio;
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
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * A compact, end-to-end FRTB SA demonstration around NablaTensor's expensive
 * calculation: full shocked repricing. Tables and controls are demo-local so
 * the compute library does not become a regulatory data-management system.
 */
public final class FrtbFullShowcase {

  private static final String PARAMETER_VERSION = "DEMO-MAR21-2026-09";
  private static final String MARKET_CSV = """
      name,value
      SPX,100.0
      USD-OIS-5Y,0.042
      ACME-SPREAD-5Y,0.012
      EURUSD,1.085
      WTI-1Y,76.5
      IMPLIED-VOL,0.20
      """;
  private static final String TRADES_CSV = """
      id,nettingSet,notional,side
      ASIAN-001,NS-OPTIONS,12000000,-1
      ASIAN-HEDGE,NS-OPTIONS,4500000,1
      MACRO-HEDGE,NS-MACRO,8000000,1
      """;

  private FrtbFullShowcase() {
  }

  public static ParameterSet parameters() {
    return ParameterSet.demoMar21();
  }

  public static MarketData marketData() {
    return MarketData.load(MARKET_CSV);
  }

  public static List<TradeSpec> trades() {
    return TradeSpec.load(TRADES_CSV);
  }

  public static void main(String[] args) {
    String engine = System.getProperty("engine", "cpu-jit");
    long scenarios = Long.getLong("scenarios", 250_000L);
    int steps = Integer.getInteger("steps", 252);
    long seed = Long.getLong("seed", 42L);

    ParameterSet parameters = parameters();
    MarketData marketData = marketData();
    List<TradeSpec> tradeSpecs = trades();
    printInputs(parameters, marketData, tradeSpecs);
    CurvatureRun heavy = runHeavyCurvature(engine, scenarios, steps, seed);
    printCurvature(heavy);
    Portfolio portfolio = buildPortfolio(tradeSpecs, parameters, marketData, heavy.cvr());
    Sensitivities netted = portfolio.aggregate();
    printNetting(portfolio, netted);
    Capital capital = aggregate(parameters, netted);
    printAggregation(capital);
    DrcResult drc = defaultRiskCharge(defaultPositions());
    RraoResult rrao = residualRiskAddOn(residualPositions(tradeSpecs));
    printAddOns(drc, rrao);
    SignOff signOff = SignOff.review(parameters, marketData, portfolio, capital, drc, rrao);
    printCapitalAndSignOff(parameters, marketData, capital, drc, rrao, signOff);
  }

  public static CurvatureRun runHeavyCurvature(String engine, long scenarios,
                                               int steps, long seed) {
    return runHeavyCurvature(engine, engine.equals("cuda"), scenarios, steps, seed);
  }

  public static CurvatureRun runHeavyCurvature(String engine, boolean fp32, long scenarios,
                                               int steps, long seed) {
    EquityMarket market = EquityMarket.atmOneYear();
    ScenarioSet shocks = ScenarioSet.list(
        Scenario.of("base"),
        Scenario.of("up", Shock.relative("spot", 0.30)),
        Scenario.of("down", Shock.relative("spot", -0.30)));

    var greekBuild = MonteCarlo.of(Products.asianCall()).market(market).steps(steps);
    var priceBuild = MonteCarlo.of(Products.asianCall()).market(market).steps(steps);
    MonteCarlo<EquityMarket> greeks = fp32
        ? greekBuild.fp32().greeks().on(engine).build()
        : greekBuild.fp64().greeks().on(engine).build();
    MonteCarlo<EquityMarket> pricer = fp32
        ? priceBuild.fp32().priceOnly().on(engine).build()
        : priceBuild.fp64().priceOnly().on(engine).build();

    try (greeks; pricer) {
      long warmup = Math.min(scenarios, 25_000L);
      greeks.run(warmup, seed);
      pricer.run(warmup, seed);
      Nabla.TypedValuation<EquityMarket> risk = greeks.run(market, scenarios, seed);
      Map<String, Nabla.TypedValuation<EquityMarket>> pv =
          ScenarioRunner.run(pricer, market, shocks, scenarios, seed);
      double position = -1.0;
      double base = position * pv.get("base").price();
      double up = position * pv.get("up").price();
      double down = position * pv.get("down").price();
      double delta = position * risk.greek(EquityMarket::spot);
      double shock = 0.30 * market.spot();
      double upResidual = up - base - shock * delta;
      double downResidual = down - base + shock * delta;
      double cvr = -Math.min(upResidual, downResidual);
      return CurvatureRun.of().engine(engine).scenarios(scenarios).steps(steps).base(base).up(up).down(down).delta(delta).shock(shock).upResidual(upResidual).downResidual(downResidual).cvr(cvr).adjointSeconds(risk.seconds()).baseSeconds(pv.get("base").seconds()).upSeconds(pv.get("up").seconds()).downSeconds(pv.get("down").seconds()).build();
    }
  }

  public static Portfolio buildPortfolio(List<TradeSpec> specs, ParameterSet parameters,
                                         MarketData marketData, double heavyCvr) {
    List<Portfolio.Trade> trades = new ArrayList<>();
    for (int tradeIndex = 0; tradeIndex < specs.size(); tradeIndex++) {
      TradeSpec spec = specs.get(tradeIndex);
      Sensitivities.Builder risk = Sensitivities.builder();
      int factorIndex = 0;
      for (var classEntry : parameters.tables().entrySet()) {
        for (String bucket : classEntry.getValue().keySet()) {
          double scale = spec.side() * spec.notional() / 1_000_000.0;
          double level = marketData.levelFor(classEntry.getKey());
          double alternating = ((factorIndex++ + tradeIndex) & 1) == 0 ? 1.0 : -0.55;
          String name = classEntry.getKey() + "-FACTOR-" + bucket;
          risk.add(RiskFactor.of().riskClass(classEntry.getKey()).measure(RiskMeasureEnum.DELTA).bucket(bucket).name(name).tenor(0.0).tenor2(0.0).build(),
              scale * level * alternating);
          risk.add(RiskFactor.of().riskClass(classEntry.getKey()).measure(RiskMeasureEnum.VEGA).bucket(bucket).name(name).tenor(1.0).tenor2(0.0).build(),
              scale * marketData.levels().get("IMPLIED-VOL") * (1.0 + tradeIndex * 0.1));
          risk.add(RiskFactor.of().riskClass(classEntry.getKey()).measure(RiskMeasureEnum.CURVATURE).bucket(bucket).name(name).tenor(0.0).tenor2(0.0).build(),
              Math.abs(scale) * 0.02 * alternating);
        }
      }
      if (tradeIndex == 0) {
        risk.add(RiskFactor.equityDelta("5", "ASIAN-001").asCurvature(), heavyCvr);
      }
      trades.add(Portfolio.trade(spec.id(), spec.nettingSet(), risk.build()));
    }
    return new Portfolio(trades);
  }

  public static Capital aggregate(ParameterSet parameters, Sensitivities sensitivities) {
    Map<RiskClassEnum, ClassCapital> byClass = new EnumMap<>(RiskClassEnum.class);
    for (RiskClassEnum riskClass : RiskClassEnum.values()) {
      Map<String, BucketParameter> table = parameters.tables().get(riskClass);
      Sensitivities classRisk = sensitivities.ofClass(riskClass);
      byClass.put(riskClass, new ClassCapital(
        aggregateMeasure(table, classRisk.ofMeasure(RiskMeasureEnum.DELTA), RiskMeasureEnum.DELTA),
        aggregateMeasure(table, classRisk.ofMeasure(RiskMeasureEnum.VEGA), RiskMeasureEnum.VEGA),
        aggregateMeasure(table, classRisk.ofMeasure(RiskMeasureEnum.CURVATURE),
          RiskMeasureEnum.CURVATURE)));
    }
    return new Capital(byClass);
  }

  private static MeasureCapital aggregateMeasure(Map<String, BucketParameter> table,
                                                 Sensitivities sensitivities,
                                                 RiskMeasureEnum measure) {
    Map<CorrelationScenarioEnum, Double> charges = new EnumMap<>(CorrelationScenarioEnum.class);
    for (CorrelationScenarioEnum scenario : CorrelationScenarioEnum.values()) {
      NestedAggregation aggregation = measure == RiskMeasureEnum.CURVATURE
        ? NestedAggregation.curvature(
          (left, right) -> left.equals(right) ? 1.0
            : scenario.apply(table.get(left.bucket()).rho()),
          (left, right) -> left.equals(right) ? 1.0
            : scenario.apply(classGamma(table)))
        : NestedAggregation.delta(
          factor -> measure == RiskMeasureEnum.DELTA
            ? table.get(factor.bucket()).deltaRw()
            : table.get(factor.bucket()).vegaRw(),
          (left, right) -> left.equals(right) ? 1.0
            : scenario.apply(table.get(left.bucket()).rho()),
          (left, right) -> left.equals(right) ? 1.0
            : scenario.apply(classGamma(table)));
      charges.put(scenario, aggregation.aggregate(sensitivities).total());
    }
    CorrelationScenarioEnum selected = charges.entrySet().stream()
      .max(Map.Entry.comparingByValue()).orElseThrow().getKey();
    return new MeasureCapital(charges, selected);
    }

  private static double classGamma(Map<String, BucketParameter> table) {
    return table.values().iterator().next().gamma();
  }

  public static DrcResult defaultRiskCharge(List<DefaultPosition> positions) {
    Map<String, Double> netByObligor = new LinkedHashMap<>();
    positions.forEach(position -> netByObligor.merge(
        position.obligor(), position.jumpToDefault(), Double::sum));
    double longs = 0.0;
    double shorts = 0.0;
    double weightedLongs = 0.0;
    double weightedShorts = 0.0;
    for (var entry : netByObligor.entrySet()) {
      DefaultPosition reference = positions.stream()
          .filter(position -> position.obligor().equals(entry.getKey())).findFirst().orElseThrow();
      double jumpToDefaultMillions = entry.getValue() / 1_000_000.0;
      if (jumpToDefaultMillions >= 0.0) {
        longs += jumpToDefaultMillions;
        weightedLongs += reference.riskWeight() * jumpToDefaultMillions;
      } else {
        shorts -= jumpToDefaultMillions;
        weightedShorts -= reference.riskWeight() * jumpToDefaultMillions;
      }
    }
    double hbr = longs + shorts == 0.0 ? 0.0 : longs / (longs + shorts);
    return new DrcResult(Math.max(weightedLongs - hbr * weightedShorts, 0.0), hbr,
      longs, shorts, weightedLongs, weightedShorts);
  }

  public static RraoResult residualRiskAddOn(List<ResidualPosition> positions) {
    double exotic = positions.stream().filter(position -> position.exotic())
      .mapToDouble(position -> Math.abs(position.grossNotional()) / 1_000_000.0 * 0.01).sum();
    double other = positions.stream().filter(position -> !position.exotic())
      .mapToDouble(position -> Math.abs(position.grossNotional()) / 1_000_000.0 * 0.001).sum();
    return new RraoResult(exotic, other);
  }

  public static List<DefaultPosition> defaultPositions() {
    return List.of(
        new DefaultPosition("ACME", 3_500_000, 0.03),
        new DefaultPosition("ACME", -1_100_000, 0.03),
        new DefaultPosition("SMALLCO", -200_000, 0.15),
        new DefaultPosition("SOVEREIGN", 2_000_000, 0.005));
  }

  public static List<ResidualPosition> residualPositions(List<TradeSpec> trades) {
    return List.of(
        new ResidualPosition(trades.get(0).id(), trades.get(0).notional(), false),
        new ResidualPosition("WEATHER-OPTION", 2_000_000, true));
  }

    private static void printInputs(ParameterSet parameters, MarketData marketData,
                    List<TradeSpec> trades) {
    System.out.printf(Locale.ROOT, "FRTB FULL  |  %s  |  market %s%n%n",
      PARAMETER_VERSION, marketData.asOf());
    System.out.println("[1/8] LOAD MARKET DATA AND TRADES");
    marketData.levels().forEach((name, value) ->
      System.out.printf(Locale.ROOT, "      market  %-18s %10.5f%n", name, value));
    for (TradeSpec trade : trades) {
      System.out.printf(Locale.ROOT, "      trade   %-12s %-12s side=%+2.0f notional=$%.1fm%n",
        trade.id(), trade.nettingSet(), trade.side(), trade.notional() / 1_000_000.0);
    }

    System.out.printf(Locale.ROOT, "%n[2/8] LOAD REGULATORY PARAMETER TABLES (%d BUCKETS)%n",
      parameters.bucketCount());
    for (RiskClassEnum riskClass : RiskClassEnum.values()) {
      Map<String, BucketParameter> table = parameters.tables().get(riskClass);
      System.out.printf(Locale.ROOT, "      %-15s %2d buckets%n", riskClass, table.size());
      printWeights(table);
      BucketParameter representative = table.values().iterator().next();
      System.out.printf(Locale.ROOT, "                      vega RW=%.3f  rho=%.3f  gamma=%.3f%n",
        representative.vegaRw(), representative.rho(), representative.gamma());
    }
    System.out.println("      DEMO tables are illustrative; a real run requires rulebook validation.");
    }

    private static void printWeights(Map<String, BucketParameter> table) {
      StringBuilder line = new StringBuilder("                      delta RW  ");
    table.forEach((bucket, parameter) -> {
        String cell = bucket + '=' + String.format(Locale.ROOT, "%.3f", parameter.deltaRw()) + ' ';
        if (line.length() + cell.length() > 72) {
          System.out.println(line);
          line.setLength(0);
          line.append("                                ");
      }
        line.append(cell);
    });
      System.out.println(line);
    }

    private static void printCurvature(CurvatureRun heavy) {
    System.out.printf(Locale.ROOT, "%n[3/8] GENERATE ADJOINT DELTA + FULL SHOCKED REPRICINGS ON %s%n",
      heavy.engine().toUpperCase(Locale.ROOT));
    System.out.printf(Locale.ROOT, "      %,d paths x %d fixings, common seed%n",
      heavy.scenarios(), heavy.steps());
    System.out.printf(Locale.ROOT, "      base PV=%10.6f   up PV=%10.6f   down PV=%10.6f%n",
      heavy.base(), heavy.up(), heavy.down());
    System.out.printf(Locale.ROOT, "      delta  =%10.6f   shock=%10.6f%n", heavy.delta(), heavy.shock());
    System.out.printf(Locale.ROOT, "      up residual   = up - base - shock*delta = %10.6f%n",
      heavy.upResidual());
    System.out.printf(Locale.ROOT, "      down residual = down - base + shock*delta = %10.6f%n",
      heavy.downResidual());
    System.out.printf(Locale.ROOT, "      CVR = -min(up residual, down residual)   = %10.6f%n",
      heavy.cvr());
    System.out.printf(Locale.ROOT,
      "      timing: adjoint=%.3f s  base=%.3f s  up=%.3f s  down=%.3f s%n",
      heavy.adjointSeconds(), heavy.baseSeconds(), heavy.upSeconds(), heavy.downSeconds());
    System.out.printf(Locale.ROOT, "      total engine work %.3f s; this is the expensive stage.%n",
      heavy.seconds());
    }

    private static void printNetting(Portfolio portfolio, Sensitivities netted) {
    System.out.println("\n[4/8] MAP TRADES TO RISK FACTORS, THEN NET");
      int observations = portfolio.trades().stream()
        .mapToInt(trade -> trade.sensitivities().asMap().size()).sum();
      System.out.printf(Locale.ROOT, "      %d trade-level factor observations enter netting%n",
        observations);
    for (var entry : portfolio.byNettingSet().entrySet()) {
      long trades = portfolio.trades().stream()
        .filter(trade -> trade.nettingSet().equals(entry.getKey())).count();
      System.out.printf(Locale.ROOT, "      %-12s %d trades -> %d net risk factors%n",
        entry.getKey(), trades, entry.getValue().asMap().size());
    }
    System.out.printf(Locale.ROOT, "      BOOK         %d trades -> %d shared net risk factors%n",
      portfolio.trades().size(), netted.asMap().size());
    System.out.println("      Asian CUDA CVR enters EQUITY bucket 5; the other demo vectors enter here.");
    System.out.println("\n      class              factors   raw delta    raw vega     raw CVR");
    for (RiskClassEnum riskClass : RiskClassEnum.values()) {
      Sensitivities classRisk = netted.ofClass(riskClass);
      System.out.printf(Locale.ROOT, "      %-18s %4d %12.4f %12.4f %12.4f%n",
        riskClass, classRisk.asMap().size(), sum(classRisk, RiskMeasureEnum.DELTA),
        sum(classRisk, RiskMeasureEnum.VEGA), sum(classRisk, RiskMeasureEnum.CURVATURE));
    }
    }

    private static double sum(Sensitivities sensitivities, RiskMeasureEnum measure) {
    return sensitivities.ofMeasure(measure).asMap().values().stream()
      .mapToDouble(Double::doubleValue).sum();
    }

    private static void printAggregation(Capital capital) {
    System.out.println("\n[5/8] APPLY RW, BUCKET CORRELATIONS, AND LOW/MEDIUM/HIGH SCENARIOS");
    System.out.println("      WS=RW*s -> Kb within bucket -> Sb clamp -> class charge -> max scenario");
    for (RiskClassEnum riskClass : RiskClassEnum.values()) {
      ClassCapital value = capital.byClass().get(riskClass);
      System.out.printf(Locale.ROOT, "      %s%n", riskClass);
      printMeasure("delta", value.delta());
      printMeasure("vega", value.vega());
      printMeasure("curvature", value.curvature());
      System.out.printf(Locale.ROOT, "          selected SBM subtotal                         %10.4f%n",
        value.total());
    }
    }

    private static void printMeasure(String label, MeasureCapital capital) {
    System.out.printf(Locale.ROOT,
          "      %-9s L=%8.4f M=%8.4f H=%8.4f -> %-4s %8.4f%n",
      label,
      capital.scenarios().get(CorrelationScenarioEnum.LOW),
      capital.scenarios().get(CorrelationScenarioEnum.MEDIUM),
      capital.scenarios().get(CorrelationScenarioEnum.HIGH),
      capital.selected(), capital.total());
    }

    private static void printAddOns(DrcResult drc, RraoResult rrao) {
    System.out.println("\n[6/8] DEFAULT RISK CHARGE");
    System.out.printf(Locale.ROOT, "      net JTD long=%.4f  short=%.4f  HBR=long/(long+short)=%.4f%n",
      drc.longs(), drc.shorts(), drc.hbr());
    System.out.printf(Locale.ROOT,
      "      DRC=max(weighted long %.4f - HBR*weighted short %.4f, 0) = %.4f%n",
      drc.weightedLongs(), drc.weightedShorts(), drc.total());
    System.out.println("\n[7/8] RESIDUAL RISK ADD-ON");
    System.out.printf(Locale.ROOT, "      exotic gross notional * 1.0%% = %.4f%n", rrao.exotic());
    System.out.printf(Locale.ROOT, "      other gross notional  * 0.1%% = %.4f%n", rrao.other());
    System.out.printf(Locale.ROOT, "      RRAO                              = %.4f%n", rrao.total());
    }

    private static void printCapitalAndSignOff(ParameterSet parameters, MarketData marketData,
                         Capital capital, DrcResult drc, RraoResult rrao,
                         SignOff signOff) {
    System.out.println("\n[8/8] CAPITAL BRIDGE, REPORT, CONTROLS, AND SIGN-OFF");
    System.out.printf(Locale.ROOT, "      SBM  = %10.4f $m%n", capital.total());
    System.out.printf(Locale.ROOT, "      DRC  = %10.4f $m%n", drc.total());
    System.out.printf(Locale.ROOT, "      RRAO = %10.4f $m%n", rrao.total());
    System.out.printf(Locale.ROOT, "      TOTAL= %10.4f $m%n",
      capital.total() + drc.total() + rrao.total());
    System.out.printf(Locale.ROOT, "      report: market=%s  parameters=%s  buckets=%d%n",
      marketData.asOf(), parameters.version(), parameters.bucketCount());
    signOff.controls().forEach(control -> System.out.println("      [PASS] " + control));
    System.out.printf(Locale.ROOT, "      sign-off: %s%n", signOff.status());
    signOff.approvers().forEach((role, name) ->
      System.out.printf(Locale.ROOT, "                %-18s %s%n", role, name));
    System.out.println("      Educational evidence package; not a regulatory filing.");
  }

  public static final class CurvatureRun {
    private final String engine;
    private final long scenarios;
    private final int steps;
    private final double base;
    private final double up;
    private final double down;
    private final double delta;
    private final double shock;
    private final double upResidual;
    private final double downResidual;
    private final double cvr;
    private final double adjointSeconds;
    private final double baseSeconds;
    private final double upSeconds;
    private final double downSeconds;

    private CurvatureRun(String engine, long scenarios, int steps, double base, double up, double down, double delta, double shock, double upResidual, double downResidual, double cvr, double adjointSeconds, double baseSeconds, double upSeconds, double downSeconds) {
      this.engine = engine;
      this.scenarios = scenarios;
      this.steps = steps;
      this.base = base;
      this.up = up;
      this.down = down;
      this.delta = delta;
      this.shock = shock;
      this.upResidual = upResidual;
      this.downResidual = downResidual;
      this.cvr = cvr;
      this.adjointSeconds = adjointSeconds;
      this.baseSeconds = baseSeconds;
      this.upSeconds = upSeconds;
      this.downSeconds = downSeconds;
    }

    public static Builder of() { return new Builder(); }

    public String engine() { return engine; }

    public long scenarios() { return scenarios; }

    public int steps() { return steps; }

    public double base() { return base; }

    public double up() { return up; }

    public double down() { return down; }

    public double delta() { return delta; }

    public double shock() { return shock; }

    public double upResidual() { return upResidual; }

    public double downResidual() { return downResidual; }

    public double cvr() { return cvr; }

    public double adjointSeconds() { return adjointSeconds; }

    public double baseSeconds() { return baseSeconds; }

    public double upSeconds() { return upSeconds; }

    public double downSeconds() { return downSeconds; }

    public double seconds() {
      return adjointSeconds + baseSeconds + upSeconds + downSeconds;
    }

    public static final class Builder {
      private String engine;
      private boolean engineSet;
      private long scenarios;
      private boolean scenariosSet;
      private int steps;
      private boolean stepsSet;
      private double base;
      private boolean baseSet;
      private double up;
      private boolean upSet;
      private double down;
      private boolean downSet;
      private double delta;
      private boolean deltaSet;
      private double shock;
      private boolean shockSet;
      private double upResidual;
      private boolean upResidualSet;
      private double downResidual;
      private boolean downResidualSet;
      private double cvr;
      private boolean cvrSet;
      private double adjointSeconds;
      private boolean adjointSecondsSet;
      private double baseSeconds;
      private boolean baseSecondsSet;
      private double upSeconds;
      private boolean upSecondsSet;
      private double downSeconds;
      private boolean downSecondsSet;

      public Builder engine(String value) {
        this.engine = value;
        this.engineSet = true;
        return this;
      }

      public Builder scenarios(long value) {
        this.scenarios = value;
        this.scenariosSet = true;
        return this;
      }

      public Builder steps(int value) {
        this.steps = value;
        this.stepsSet = true;
        return this;
      }

      public Builder base(double value) {
        this.base = value;
        this.baseSet = true;
        return this;
      }

      public Builder up(double value) {
        this.up = value;
        this.upSet = true;
        return this;
      }

      public Builder down(double value) {
        this.down = value;
        this.downSet = true;
        return this;
      }

      public Builder delta(double value) {
        this.delta = value;
        this.deltaSet = true;
        return this;
      }

      public Builder shock(double value) {
        this.shock = value;
        this.shockSet = true;
        return this;
      }

      public Builder upResidual(double value) {
        this.upResidual = value;
        this.upResidualSet = true;
        return this;
      }

      public Builder downResidual(double value) {
        this.downResidual = value;
        this.downResidualSet = true;
        return this;
      }

      public Builder cvr(double value) {
        this.cvr = value;
        this.cvrSet = true;
        return this;
      }

      public Builder adjointSeconds(double value) {
        this.adjointSeconds = value;
        this.adjointSecondsSet = true;
        return this;
      }

      public Builder baseSeconds(double value) {
        this.baseSeconds = value;
        this.baseSecondsSet = true;
        return this;
      }

      public Builder upSeconds(double value) {
        this.upSeconds = value;
        this.upSecondsSet = true;
        return this;
      }

      public Builder downSeconds(double value) {
        this.downSeconds = value;
        this.downSecondsSet = true;
        return this;
      }

      public Builder from(CurvatureRun value) {
        if (value == null) throw new NullPointerException("value");
        engine(value.engine());
        scenarios(value.scenarios());
        steps(value.steps());
        base(value.base());
        up(value.up());
        down(value.down());
        delta(value.delta());
        shock(value.shock());
        upResidual(value.upResidual());
        downResidual(value.downResidual());
        cvr(value.cvr());
        adjointSeconds(value.adjointSeconds());
        baseSeconds(value.baseSeconds());
        upSeconds(value.upSeconds());
        downSeconds(value.downSeconds());
        return this;
      }

      public CurvatureRun build() {
        if (!engineSet) throw new IllegalStateException("Missing required value: engine");
        if (!scenariosSet) throw new IllegalStateException("Missing required value: scenarios");
        if (!stepsSet) throw new IllegalStateException("Missing required value: steps");
        if (!baseSet) throw new IllegalStateException("Missing required value: base");
        if (!upSet) throw new IllegalStateException("Missing required value: up");
        if (!downSet) throw new IllegalStateException("Missing required value: down");
        if (!deltaSet) throw new IllegalStateException("Missing required value: delta");
        if (!shockSet) throw new IllegalStateException("Missing required value: shock");
        if (!upResidualSet) throw new IllegalStateException("Missing required value: upResidual");
        if (!downResidualSet) throw new IllegalStateException("Missing required value: downResidual");
        if (!cvrSet) throw new IllegalStateException("Missing required value: cvr");
        if (!adjointSecondsSet) throw new IllegalStateException("Missing required value: adjointSeconds");
        if (!baseSecondsSet) throw new IllegalStateException("Missing required value: baseSeconds");
        if (!upSecondsSet) throw new IllegalStateException("Missing required value: upSeconds");
        if (!downSecondsSet) throw new IllegalStateException("Missing required value: downSeconds");
        return new CurvatureRun(engine, scenarios, steps, base, up, down, delta, shock, upResidual, downResidual, cvr, adjointSeconds, baseSeconds, upSeconds, downSeconds);
      }
    }
  }

  public record BucketParameter(double deltaRw, double vegaRw, double rho, double gamma) {
  }

  public record ParameterSet(String version,
                             Map<RiskClassEnum, Map<String, BucketParameter>> tables) {
    public ParameterSet {
      EnumMap<RiskClassEnum, Map<String, BucketParameter>> copy = new EnumMap<>(RiskClassEnum.class);
      tables.forEach((riskClass, table) -> copy.put(riskClass,
          Collections.unmodifiableMap(new LinkedHashMap<>(table))));
      tables = Collections.unmodifiableMap(copy);
      if (tables.size() != RiskClassEnum.values().length) {
        throw new IllegalArgumentException("a table is required for every risk class");
      }
    }

    public int bucketCount() {
      return tables.values().stream().mapToInt(Map::size).sum();
    }

    public static ParameterSet demoMar21() {
      EnumMap<RiskClassEnum, Map<String, BucketParameter>> tables = new EnumMap<>(RiskClassEnum.class);
      tables.put(RiskClassEnum.GIRR, named(new String[] {"USD", "EUR", "JPY"},
          new double[] {0.017, 0.017, 0.017}, 1.0, 0.40, 0.50));
      tables.put(RiskClassEnum.CSR_NON_SEC, numbered(new double[] {
          0.005, 0.010, 0.050, 0.030, 0.030, 0.020, 0.015, 0.025, 0.020,
          0.040, 0.120, 0.070, 0.085, 0.055, 0.050, 0.120, 0.015, 0.050
      }, 1.0, 0.35, 0.40));
      tables.put(RiskClassEnum.CSR_SEC, interpolated(25, 0.009, 0.075, 1.0, 0.40, 0.40));
      tables.put(RiskClassEnum.CSR_SEC_CTP, interpolated(16, 0.040, 0.130, 1.0, 0.35, 0.40));
      tables.put(RiskClassEnum.EQUITY, numbered(new double[] {
          0.55, 0.60, 0.45, 0.55, 0.30, 0.35, 0.40,
          0.50, 0.70, 0.50, 0.70, 0.15, 0.25
      }, 1.0, 0.25, 0.15));
      tables.put(RiskClassEnum.COMMODITY, numbered(new double[] {
          0.30, 0.35, 0.60, 0.80, 0.40, 0.45, 0.20, 0.35, 0.25, 0.35, 0.50
      }, 1.0, 0.55, 0.20));
      tables.put(RiskClassEnum.FX, named(new String[] {"EURUSD", "USDJPY", "GBPUSD"},
          new double[] {0.15, 0.15, 0.15}, 1.0, 1.0, 0.60));
      return new ParameterSet(PARAMETER_VERSION, tables);
    }

    private static Map<String, BucketParameter> numbered(
        double[] weights, double vega, double rho, double gamma) {
      String[] names = new String[weights.length];
      for (int index = 0; index < names.length; index++) {
        names[index] = Integer.toString(index + 1);
      }
      return named(names, weights, vega, rho, gamma);
    }

    private static Map<String, BucketParameter> interpolated(
        int count, double low, double high, double vega, double rho, double gamma) {
      double[] weights = new double[count];
      for (int index = 0; index < count; index++) {
        weights[index] = low + (high - low) * index / Math.max(1, count - 1);
      }
      return numbered(weights, vega, rho, gamma);
    }

    private static Map<String, BucketParameter> named(
        String[] names, double[] weights, double vega, double rho, double gamma) {
      Map<String, BucketParameter> table = new LinkedHashMap<>();
      for (int index = 0; index < names.length; index++) {
        table.put(names[index], new BucketParameter(weights[index], vega, rho, gamma));
      }
      return table;
    }
  }

  public record MarketData(LocalDate asOf, Map<String, Double> levels) {
    public MarketData {
      levels = Map.copyOf(levels);
    }

    public static MarketData load(String csv) {
      return new MarketData(LocalDate.of(2026, 9, 2), parseCsv(csv));
    }

    public double levelFor(RiskClassEnum riskClass) {
      return switch (riskClass) {
        case GIRR -> levels.get("USD-OIS-5Y");
        case CSR_NON_SEC, CSR_SEC, CSR_SEC_CTP -> levels.get("ACME-SPREAD-5Y");
        case EQUITY -> levels.get("SPX") / 100.0;
        case COMMODITY -> levels.get("WTI-1Y") / 100.0;
        case FX -> levels.get("EURUSD");
      };
    }
  }

  public record TradeSpec(String id, String nettingSet, double notional, double side) {
    public static List<TradeSpec> load(String csv) {
      List<String[]> rows = parseRows(csv);
      return rows.stream().map(row -> new TradeSpec(
          row[0], row[1], Double.parseDouble(row[2]), Double.parseDouble(row[3]))).toList();
    }
  }

  public record MeasureCapital(Map<CorrelationScenarioEnum, Double> scenarios,
                               CorrelationScenarioEnum selected) {
    public MeasureCapital {
      scenarios = Map.copyOf(scenarios);
    }

    public double total() {
      return scenarios.get(selected);
    }
  }

  public record ClassCapital(MeasureCapital delta, MeasureCapital vega,
                             MeasureCapital curvature) {
    public double total() {
      return delta.total() + vega.total() + curvature.total();
    }
  }

  public record Capital(Map<RiskClassEnum, ClassCapital> byClass) {
    public Capital {
      EnumMap<RiskClassEnum, ClassCapital> copy = new EnumMap<>(RiskClassEnum.class);
      copy.putAll(byClass);
      byClass = Collections.unmodifiableMap(copy);
    }

    public double total() {
      return byClass.values().stream().mapToDouble(ClassCapital::total).sum();
    }
  }

  public record DefaultPosition(String obligor, double jumpToDefault, double riskWeight) {
  }

  public record DrcResult(double total, double hbr, double longs, double shorts,
                          double weightedLongs, double weightedShorts) {
  }

  public record ResidualPosition(String id, double grossNotional, boolean exotic) {
  }

  public record RraoResult(double exotic, double other) {
    public double total() {
      return exotic + other;
    }
  }

  public record SignOff(String status, List<String> controls, Map<String, String> approvers) {
    public static SignOff review(ParameterSet parameters, MarketData marketData,
                                 Portfolio portfolio, Capital capital,
                                 DrcResult drc, RraoResult rrao) {
      boolean complete = parameters.tables().size() == RiskClassEnum.values().length
          && !marketData.levels().isEmpty()
          && !portfolio.trades().isEmpty()
          && capital.byClass().size() == RiskClassEnum.values().length
          && drc.total() >= 0.0 && rrao.total() >= 0.0;
      List<String> controls = List.of(
          "market snapshot is dated and non-empty",
          "all seven parameter tables and 89 buckets are present",
          "trade and netting-set lineage is retained",
          "LOW/MEDIUM/HIGH results exist for delta, vega, and curvature",
          "DRC and RRAO are non-negative");
      Map<String, String> approvers = new LinkedHashMap<>();
      approvers.put("Model owner", "A. Quant (demo)");
      approvers.put("Risk control", "B. Controller (demo)");
      approvers.put("Regulatory owner", "C. Officer (demo)");
      return new SignOff(complete ? "APPROVED" : "REJECTED",
          controls, Collections.unmodifiableMap(approvers));
    }
  }

  private static Map<String, Double> parseCsv(String csv) {
    Map<String, Double> values = new LinkedHashMap<>();
    for (String[] row : parseRows(csv)) {
      values.put(row[0], Double.parseDouble(row[1]));
    }
    return values;
  }

  private static List<String[]> parseRows(String csv) {
    try (BufferedReader reader = new BufferedReader(new StringReader(csv))) {
      return reader.lines().skip(1).filter(line -> !line.isBlank())
          .map(line -> line.split(",", -1)).toList();
    } catch (IOException exception) {
      throw new IllegalStateException("cannot load embedded demo CSV", exception);
    }
  }
}