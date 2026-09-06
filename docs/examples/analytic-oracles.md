# Analytic pricer oracles for adjoint Monte-Carlo

*Keywords: black scholes closed form java, black 76 java, bachelier normal model java, merton jump diffusion java, reiner rubinstein barrier java, cds par spread java*

The adjoint Monte-Carlo engine is checked two ways: against the scalar CPU
oracle path-for-path, and against a **numerical** bump of the same estimator.
Feature **F2** adds the third and strongest check where it exists — an
independent **closed form** that shares no code with the simulation.

Source: [`nablatensor-quant/.../analytic/`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/analytic/)
· example [`AnalyticVsAdjoint.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/AnalyticVsAdjoint.java)

## What's in the package

| Class | Model | Typical use |
|---|---|---|
| `CostOfCarry` | generalised Black-Scholes-Merton with carry `b` | the shared core of the vanilla family |
| `Black76` | option on a forward / futures price (`b = 0`) | caps, floors, European swaptions, futures options |
| `GeneralizedBsm` | continuous dividend yield `q` (`b = r - q`) | stock and index options; exposes `dividendRho` |
| `GarmanKohlhagen` | FX, foreign rate `r_f` (`b = r_d - r_f`) | FX options; exposes `foreignRho` |
| `Bachelier` | normal (arithmetic) forward | rate options in a low- or negative-rate regime |
| `MertonJumpDiffusion` | diffusion + lognormal compound-Poisson jumps | the oracle for the F7 jump-diffusion step block |
| `BarrierAnalytic` | Reiner-Rubinstein, continuous monitoring, zero rebate | the reference for the smoothed `ExoticProducts.barrier` |
| `CdsParSpread` | survival curve + discount curve on a grid | the reference for bootstrapped-hazard CDS pricing |

Every vanilla-family pricer returns an `AnalyticGreeks` record: the **price in
closed form**, and `delta / gamma / vega / theta / rho / dV/dK` by central
differencing of that closed form. The differences carry no Monte-Carlo noise and
only `~1e-7` truncation error — writing the six Greeks this way instead of
transcribing six more formulas per model removes a class of copy error, and the
package tests still pin the differenced values against the published
Black-Scholes Greeks.

`N(x)` here is West's rational approximation (double precision), not the lighter
`erfc` in `BlackScholes` — a second difference amplifies any bias in the normal
CDF by `1/h^2`.

## Using one as a model-validation reference

```java
Report report = ModelValidation.of(Products.europeanCall())
    .market(EquityMarket.atmOneYear()).steps(1)
    .scenarios(500_000).seed(42L)
    .analyticReference(m -> GeneralizedBsm.of(OptionType.CALL,
        m.spot(), m.strike(), m.maturity(), m.rate(), 0.0, m.vol()).greeks())
    .run();
System.out.println(report);
```

The evidence pack gains a section diffing the scalar oracle's adjoint price and
gradient against the closed form, with a `3·SE + 5e-3·|ref|` acceptance band.

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.AnalyticVsAdjoint -Dscenarios=2000000
```

## Cross-identities the tests pin

- `Black76` at `F = S e^{(r-q)T}` equals `GeneralizedBsm` — same price to `1e-10`.
- `GeneralizedBsm` with `q = 0` equals `BlackScholes` (to that class's coarser
  `~1e-6`).
- `MertonJumpDiffusion` with `lambda = 0` collapses to `GeneralizedBsm`; with
  jumps it still satisfies put-call parity `C - P = S - K e^{-rT}`.
- `Bachelier` ATM price is `sigmaN sqrt(T) / sqrt(2 pi)`; put-call parity holds
  with discounting.
- `BarrierAnalytic`: knock-in + knock-out = the vanilla, for all eight
  up/down x in/out x call/put combinations; an unreachable barrier recovers the
  vanilla.
- `CdsParSpread` with a flat hazard `lambda` and recovery `R` returns
  `~ lambda (1 - R)` (the credit triangle), and a par contract has zero value.
