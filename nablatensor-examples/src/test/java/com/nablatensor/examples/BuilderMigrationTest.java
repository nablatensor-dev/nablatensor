/*
 * Copyright 2026 The NablaTensor Authors
 * Licensed under the Apache License, Version 2.0.
 */
package com.nablatensor.examples;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/** Guards the public construction boundary established by the builder migration. */
final class BuilderMigrationTest {

  private static final String[] VALUE_TYPES = {
      "com.nablatensor.engine.AadOptions",
      "com.nablatensor.tensor.ConvSpec",
      "com.nablatensor.risk.RiskFactor",
      "com.nablatensor.scenario.Shock",
      "com.nablatensor.quant.EquityMarket",
      "com.nablatensor.quant.FxMarket",
      "com.nablatensor.quant.BasketMarket",
      "com.nablatensor.quant.SpreadMarket",
      "com.nablatensor.quant.QuantoMarket",
      "com.nablatensor.quant.HestonMarket",
      "com.nablatensor.quant.MertonJumpMarket",
      "com.nablatensor.quant.KouMarket",
      "com.nablatensor.quant.SabrMarket",
      "com.nablatensor.quant.HullWhiteMarket",
      "com.nablatensor.quant.LmmMarket",
      "com.nablatensor.quant.LocalVolMarket",
      "com.nablatensor.quant.SchwartzMarket",
      "com.nablatensor.cva.CvaMarket",
      "com.nablatensor.quant.YieldCurve",
      "com.nablatensor.quant.CurveSet",
      "com.nablatensor.credit.CreditCurve",
      "com.nablatensor.cva.CreditName",
      "com.nablatensor.cva.CdsQuote",
      "com.nablatensor.credit.CopulaMarket",
      "com.nablatensor.credit.CdoTranche",
      "com.nablatensor.quant.Seasonality",
      "com.nablatensor.quant.estimate.Garch11",
      "com.nablatensor.quant.transform.BsmCf",
      "com.nablatensor.quant.transform.HestonCf",
      "com.nablatensor.quant.transform.VarianceGammaCf",
      "com.nablatensor.cva.SaCvaParameters",
      "com.nablatensor.cva.BaCvaParameters",
      "com.nablatensor.cva.CollateralAgreement",
      "com.nablatensor.cva.NettingSet",
      "com.nablatensor.cva.InterestRateSwap",
      "com.nablatensor.cva.FxForward",
      "com.nablatensor.cva.CvaHedge",
      "com.nablatensor.cva.CvaResult",
      "com.nablatensor.risk.VarBacktest",
      "com.nablatensor.quant.analytic.AnalyticGreeks",
      "com.nablatensor.quant.analytic.GeneralizedBsm",
      "com.nablatensor.quant.analytic.GarmanKohlhagen",
      "com.nablatensor.quant.analytic.Margrabe",
      "com.nablatensor.quant.BlackScholes",
      "com.nablatensor.lattice.LatticeGreeks",
      "com.nablatensor.validate.Report",
      "com.nablatensor.quant.transform.HestonCosCalibrator$Quote",
      "com.nablatensor.quant.HullWhiteCalibration$SwaptionQuote",
      "com.nablatensor.examples.ClimateScenarioShowcase$Pathway",
      "com.nablatensor.examples.FrtbFullShowcase$CurvatureRun",
      "com.nablatensor.examples.SimmShowcase$HeavyRun",
      "com.nablatensor.examples.Fp32BackendComparisonTest$Row",
      "com.nablatensor.engine.jit.KernelGenerator$ArraySlots"
  };

  private static final String[] FACTORY_TYPES = {
      "com.nablatensor.quant.GbmPath",
      "com.nablatensor.quant.HestonModel",
      "com.nablatensor.quant.HullWhite1F",
      "com.nablatensor.quant.SabrModel",
      "com.nablatensor.quant.LmmModel",
      "com.nablatensor.quant.SchwartzOneFactor",
      "com.nablatensor.quant.LocalVolModel",
      "com.nablatensor.quant.MertonJumpModel",
      "com.nablatensor.quant.KouJumpModel",
      "com.nablatensor.cva.HwShortRate",
      "com.nablatensor.cva.ExposureSimulation",
      "com.nablatensor.lattice.BinomialTree"
  };



  private static final String[] ENUM_TYPES = {
      "com.nablatensor.engine.AadOptions$PrecisionEnum",
      "com.nablatensor.engine.JitOptimizations$CategoryEnum",
      "com.nablatensor.engine.JitOptimizations$LevelEnum",
      "com.nablatensor.engine.AadOpEnum",
      "com.nablatensor.tensor.BackendEnum",
      "com.nablatensor.tensor.DTypeEnum",
      "com.nablatensor.tensor.OpEnum",
      "com.nablatensor.tensor.DeviceTypeEnum",
      "com.nablatensor.tensor.tree.TreeDef$KindEnum",
      "com.nablatensor.scenario.Shock$KindEnum",
      "com.nablatensor.lattice.BinomialTree$MethodEnum",
      "com.nablatensor.quant.OptionTypeEnum",
      "com.nablatensor.quant.ExoticProducts$BarrierEnum",
      "com.nablatensor.quant.analytic.BarrierAnalytic$KindEnum",
      "com.nablatensor.cva.InterestRateSwap$SideEnum",
      "com.nablatensor.cva.FxForward$SideEnum",
      "com.nablatensor.cva.CvaHedge$KindEnum",
      "com.nablatensor.cva.CreditName$RatingEnum",
      "com.nablatensor.cva.CreditName$SectorEnum",
      "com.nablatensor.risk.RiskClassEnum",
      "com.nablatensor.risk.RiskMeasureEnum",
      "com.nablatensor.risk.CorrelationScenarioEnum",
      "com.nablatensor.risk.RiskFactor$CsrCurveEnum",
      "com.nablatensor.examples.SimmShowcase$ProductClassEnum",
      "com.nablatensor.validate.BumpCrossCheck$EquityFactorEnum"
  };

  private static final String[][] POSITIONAL_FACTORIES_REMOVED = {
      {"com.nablatensor.quant.ExoticProducts", ",barrier,digitalCash,digitalAsset,cliquet,autocallable,"},
      {"com.nablatensor.quant.BermudanOption", ",option,"},
      {"com.nablatensor.quant.Hooks", ",controlVariate,importanceSampling,pathFilter,"},
      {"com.nablatensor.quant.analytic.GeneralizedBsm", ",of,"},
      {"com.nablatensor.quant.analytic.GarmanKohlhagen", ",of,"},
      {"com.nablatensor.quant.analytic.Margrabe", ",of,"},
      {"com.nablatensor.quant.analytic.Black76", ",of,"},
      {"com.nablatensor.quant.analytic.Bachelier", ",of,"},
      {"com.nablatensor.quant.analytic.BarrierAnalytic", ",of,"},
      {"com.nablatensor.quant.analytic.MertonJumpDiffusion", ",of,"},
      {"com.nablatensor.cva.InterestRateSwap", ",payer,receiver,"},
      {"com.nablatensor.cva.CvaHedge", ",singleName,index,"},
      {"com.nablatensor.risk.VarBacktest", ",of,"},
      {"com.nablatensor.risk.RiskFactor", ",equityVega,equityRepoDelta,girrVega,csrDelta,csrVega,commodityDelta,commodityVega,"}
  };

  @Test
  void migratedTypesDoNotExposePositionalConstructors() {
    Stream.concat(Stream.of(VALUE_TYPES), Stream.of(FACTORY_TYPES)).map(BuilderMigrationTest::load)
        .forEach(type -> assertFalse(Stream.of(type.getDeclaredConstructors())
            .anyMatch(constructor -> Modifier.isPublic(constructor.getModifiers())), type.getName()));
  }

  @Test
  void builderValuesAreFinalClassesRatherThanRecords() {
    Stream.of(VALUE_TYPES).map(BuilderMigrationTest::load).forEach(type -> {
      assertTrue(Modifier.isFinal(type.getModifiers()), type.getName());
      assertFalse(type.isRecord(), type.getName());
    });
  }



  @Test
  void enumTypesUseTheEnumSuffix() {
    Stream.of(ENUM_TYPES).map(BuilderMigrationTest::load).forEach(type -> {
      assertTrue(type.isEnum(), type.getName());
      assertTrue(type.getSimpleName().endsWith("Enum"), type.getName());
    });
  }

  @Test
  void migratedApisDoNotExposePositionalFactoryBypasses() {
    for (String[] entry : POSITIONAL_FACTORIES_REMOVED) {
      Class<?> type = load(entry[0]);
      assertFalse(Stream.of(type.getDeclaredMethods()).anyMatch(method ->
          Modifier.isPublic(method.getModifiers()) && Modifier.isStatic(method.getModifiers())
              && method.getParameterCount() > 0 && entry[1].contains("," + method.getName() + ",")),
          type.getName());
    }
  }

  private static Class<?> load(String name) {
    try {
      return Class.forName(name);
    } catch (ClassNotFoundException exception) {
      throw new AssertionError(name, exception);
    }
  }
}
