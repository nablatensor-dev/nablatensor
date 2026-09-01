# FRTB SA — Sensitivities-Based Method

*Keywords: FRTB SA-SBM java, sensitivities based method, FRTB delta vega curvature open source, GIRR CSR equity commodity FX SBM, MAR21 capital charge*

The SBM delta and vega vectors are exactly what **one adjoint sweep of the book
produces**. `nablatensor-reg` maps them onto MAR21 risk factors and runs the
two-level aggregation for **all seven risk classes**; curvature is two shocked
repricings per factor via the [scenario DSL](../examples/scenario-dsl.md).

> **Calculators, not sign-off.** This computes the number MAR21 asks for. Model
> validation and regulatory submission are the user's. Every risk weight and
> correlation lives in a `*Parameters` class citing the Basel paragraph — check
> them against your regulator's current rulebook. CSR **securitisation** and
> **CTP** ship structurally-correct **placeholder** tables (like ISDA SIMM);
> replace them with the MAR21.66 / MAR21.74 values before real use.

## Coverage

| Risk class | Entry point | Parameters | Notes |
|---|---|---|---|
| GIRR | `sbm.girr.GirrSbm` | `GirrSbmParameters` | 10 vertices, inflation, cross-currency basis; √2 relief for liquid currencies; absolute curvature shift |
| CSR non-securitisations | `sbm.csr.CsrSbm.nonSec` | `CsrSbmParameters.nonSec()` | 18 buckets, bond/CDS basis, `ρ_name·ρ_tenor·ρ_basis` |
| CSR securitisations (non-CTP) | `sbm.csr.CsrSbm.securitisation` | `CsrSbmParameters.securitisation()` | **placeholder tables** |
| CSR securitisations (CTP) | `sbm.csr.CsrSbm.ctp` | `CsrSbmParameters.ctp()` | **placeholder tables** |
| Equity | `sbm.equity.EquitySbmProfile` / `frtb.FrtbSaSbm` | `EquitySbmParameters` | published MAR21 equity spot values; repo-rate factor added (placeholder weight) |
| Commodity | `sbm.commodity.CommoditySbm` | `CommoditySbmParameters` | 11 buckets, maturity × delivery location |
| FX | `sbm.fx.FxSbm` | `FxSbmParameters` | one factor per pair, √2 relief for liquid pairs |

The equity-only `FrtbSaSbm.equity(...)` facade stays; the generic engine
reproduces it exactly (`FrtbSaTest.genericEngineReproducesTheEquityFacade`).

## The generic engine

```java
// one RiskClassProfile carries a class's MAR21 tables; SbmCharge runs it
SbmCharge.Result r = SbmCharge.of(GirrSbmParameters.baselDefault())
    .compute(bookSensitivities, curvatureRepricings);
r.delta(); r.vega(); r.curvature(); r.total(); r.bindingScenario(); r.perScenario();
```

`bookSensitivities` is a `Sensitivities` keyed by `RiskFactor` — build the keys
with the typed factories (`RiskFactor.girrDelta(ccy, curveId, vertex)`,
`csrDelta`, `commodityDelta`, `fxDelta`, `equityRepoDelta`, ...). Curvature
inputs are `CurvatureRepricing(factor, riskFactorLevel, netDelta, pvBase, pvUp,
pvDown)` — one per curve for GIRR/CSR.

## Aggregation

Per charge and per scenario, `NestedAggregation` (unchanged from the equity
slice):

```
WS_k  = RW_k · s_k
K_b   = √(max(0, Σ WS_k² + Σ_{k≠l} ρ_kl WS_k WS_l))            (curvature: max(WS,0)², ρ², ψ gating)
S_b   = clamp(Σ WS_k, −K_b, K_b)
charge= √(max(0, Σ K_b² + Σ_{b≠c} γ_bc S_b S_c))              (curvature: γ², ψ gating)
```

Curvature: `CVR_k = −min(V(x+shock) − V(x) − shock·δ_k , V(x−shock) − V(x) + shock·δ_k)`,
`shock` from the profile (relative `RW·level` for equity/commodity/FX, absolute
`RW` for GIRR/CSR curves).

The three MAR21.6 correlation scenarios (`LOW` / `MEDIUM` / `HIGH`) are applied
by `SbmCharge`; the class charge is the largest of the three, and
`bindingScenario()` says which.

## The full SA charge

```java
FrtbSa.Result r = FrtbSa.of("EUR")
    .sbm(bookSensitivities, curvatureRepricings)   // dispatches all 7 classes
    .drc(defaultRiskPositions)                     // see frtb-sa-drc.md
    .rrao(residualPositions)                       // see frtb-sa-rrao.md
    .compute();
r.sbm(); r.drc(); r.rrao(); r.total();             // total = Σ_class max(H,M,L) + DRC + RRAO
r.perRiskClass(); r.corep();                       // COREP C 90.xx-shaped rows
FrtbSa.dual(relieved, unrelieved);                 // CRR3 relieved/unrelieved: two param sets, one book
```

## Guided tour

`nablatensor-reg/src/test/java/com/nablatensor/reg/tour/` — 20 runnable lessons,
easy to hard (`Lesson01_WeightedSensitivity` … `Lesson20_RelievedUnrelievedDual`),
each a self-contained explanation with a worked example the code reproduces.

## Verification (`SbmChargeTest`, `FrtbSaTest`)

An **independent** nested aggregation, written fresh in the test (not calling
`NestedAggregation`), reproduces the calculator's delta + vega + curvature for
**every** correlation scenario — for GIRR, CSR non-sec, commodity and FX — and
the reported total is the largest of the three. `FrtbSaTest` reconciles the
generic engine to the equity facade and checks `total = SBM + DRC + RRAO`.

## Not in this slice

CSR securitisation / CTP calibrated parameter tables; the equity repo-rate MAR21
weight table; the full MAR21.60 CSR across-bucket matrix (a simplification
ships); the prescribed-bump / adjoint **sensitivity extraction** bridge
(callers supply `Sensitivities` and repricings today); COREP XBRL binding; the
EU targeted multiplier and operational-relief switches (`FrtbSa.dual` plumbing
is here, the relief parameters are a later phase).
