# Positional-constructor migration plan

## Goal

Replace public positional construction of the listed configuration, market, trade, and model types with named construction through:

```java
Type.of()
    .field(value)
    .build();
```

For applicable immutable value types, direct public constructors must no longer be available.

## Construction contract

The internal `nablatensor-codegen` module supplies `@Of`.

A migrated value type:

1. Is a `final` class with private state.
2. Has a private validating constructor.
3. Exposes a package-private `create(...)` entry point for generated code.
4. Exposes `public static TypeBuilder of()`.
5. Keeps named accessor methods such as `spot()`.
6. Preserves defensive copying and domain validation.

Generated `TypeBuilder` classes provide one named setter per component, required-field validation (including primitive fields), `from(Type)`, and `build()`.

## AAD market compatibility

The AAD API must no longer require `M extends Record`.

`MarketShape` must support immutable `@Of` market classes with:

- non-static `double` fields in declaration order;
- public accessors with matching names;
- a private all-component constructor usable for internal Greek reconstruction.

The following APIs become generic over `M` rather than `M extends Record`:

- `Nabla.model`
- `Nabla.Inputs`
- `Nabla.TypedModel`
- `Nabla.TypedPricer`
- `Nabla.TypedRequest`
- `Nabla.TypedValuation`
- `Product`
- `MonteCarlo`

Add parity tests covering price and Greeks for an `@Of` market class.

## Migration batches

### Core and tensor

- `AadOptions`
- `ConvSpec`

### Risk and scenarios

- `RiskFactor`
- `Shock`

### AAD market types

- `EquityMarket`
- `FxMarket`
- `BasketMarket`
- `SpreadMarket`
- `QuantoMarket`
- `HestonMarket`
- `MertonJumpMarket`
- `KouMarket`
- `SabrMarket`
- `HullWhiteMarket`
- `LmmMarket`
- `LocalVolMarket`
- `SchwartzMarket`
- `CvaMarket`

### Curves, credit, and transforms

- `YieldCurve`
- `CurveSet`
- `CreditCurve`
- `CreditName`
- `CdsQuote`
- `CopulaMarket`
- `CdoTranche`
- `Seasonality`
- `Garch11`
- `BsmCf`
- `HestonCf`
- `VarianceGammaCf`

### CVA parameters and trades

- `SaCvaParameters`
- `BaCvaParameters`
- `CollateralAgreement`
- `NettingSet`
- `InterestRateSwap`
- `FxForward`
- `CvaHedge`

### Products and analytic inputs

Public APIs that construct configured payoffs or analytic results also use named builders:

- `ExoticProducts.BarrierOption`, `DigitalCash`, `DigitalAsset`, `Cliquet`, `Autocallable`
- `BermudanOption`
- `Hooks.ControlVariate`, `ImportanceSampling`, `PathFilter`
- `GeneralizedBsm`, `GarmanKohlhagen`, `Margrabe`, `Black76`, `Bachelier`
- `BarrierAnalytic`, `MertonJumpDiffusion`
- `VarBacktest.Analysis`

Named convenience factories on `InterestRateSwap` and `CvaHedge` must not bypass their generated builders.
Bare numerical functions such as `price(...)`, `blackVol(...)`, calibration operations, and runtime execution methods are calculations rather than object construction and remain methods.

### Model and execution APIs

Use named `of(...)` factories where a field builder adds no clarity:

- `GbmPath`
- `HestonModel`
- `HullWhite1F`
- `SabrModel`
- `LmmModel`
- `SchwartzOneFactor`
- `LocalVolModel`
- `MertonJumpModel`
- `KouJumpModel`
- `HwShortRate`
- `ExposureSimulation`

`GbmPath` remains subclassable; its constructors may be `protected`, while callers use `GbmPath.of(...)`.

## Per-batch completion checklist

1. Convert target types and add `@Of` where applicable.
2. Replace every production, test, example, and benchmark `new Type(...)` call.
3. Update convenience factories to use the new API.
4. Search for remaining direct construction:
   ```bash
   rg 'new Type\\(' --glob '*.java'
   ```
5. Compile the affected modules and dependents.
6. Run the affected tests.
7. Update API documentation and examples.

## Final verification

- No external direct construction remains for migrated types.
- The full Maven reactor builds and tests cleanly.
- Public examples use `of()&build()`.
- A regression check prevents reintroducing direct construction.


## Completion status

Completed on 2026-09-21.

- All listed immutable values use generated named builders with required-field tracking and `from(...)`.
- All listed model/execution types expose named `of(...)` factories; positional constructors are non-public.
- AAD typed markets accept immutable classes and reconstruct Greeks through private all-component constructors.
- Production, tests, examples, benchmarks, shell demos, Python examples, and notebooks use named construction.
- Configured payoff and analytic-result factories expose zero-argument `.of()` builders; positional overloads are non-public.
- `BuilderMarketTest` covers builder validation, `from(...)`, price, and Greek parity.
- `BuilderMigrationTest` prevents public positional constructors and record regressions.
- Verified with `mvn -q clean test` and the direct-construction search from this plan.
