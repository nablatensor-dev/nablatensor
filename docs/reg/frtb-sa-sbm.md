# FRTB SA — Sensitivities-Based Method (equity)

*Keywords: FRTB SA-SBM java, sensitivities based method equity, FRTB delta vega curvature open source, MAR21 capital charge*

The SBM delta and vega vectors are exactly what **one adjoint sweep of the book
produces**. `nablatensor-reg` maps them onto FRTB risk factors and runs the
MAR21 aggregation; curvature is two shocked repricings per name via the
[scenario DSL](../examples/scenario-dsl.md).

> **Calculators, not sign-off.** This computes the number MAR21 asks for. Model
> validation and regulatory submission are the user's. The risk weights and
> correlations in `EquitySbmParameters` are a data table from the Basel
> Framework MAR21 (BCBS d457) — check them against your regulator's current
> rulebook.

## Coverage

- **Risk class:** equity (the 13 SBM buckets), spot-price factor.
- **Charges:** delta + vega + curvature.
- **Correlation scenarios:** `LOW` / `MEDIUM` / `HIGH` (MAR21.6); the charge is
  the largest of the three, and `Result.bindingScenario()` says which.

```java
// book sensitivities from one adjoint sweep, mapped to RiskFactors
Sensitivities book = ...;   // RiskFactor.equityDelta("5","ACME") -> dPV/dSpot, equityVega(...) -> dPV/dVol

// curvature: PV, PV(spot*(1+RW)), PV(spot*(1-RW)) per name  (RW from EquitySbmParameters)
List<FrtbSaSbm.CurvatureInput> curv = ...;

FrtbSaSbm.Result r = FrtbSaSbm.equity(book, curv);
r.delta(); r.vega(); r.curvature(); r.total(); r.bindingScenario();
```

## Aggregation

Per charge and per scenario, `NestedAggregation`:

```
WS_k  = RW_bucket · s_k
K_b   = √(max(0, Σ WS_k² + Σ_{k≠l} ρ_kl WS_k WS_l))            (curvature: max(WS,0)², ρ², ψ gating)
S_b   = max(min(Σ WS_k, K_b), −K_b)
charge= √(max(0, Σ K_b² + Σ_{b≠c} γ_bc S_b S_c))              (curvature: γ², ψ gating)
```

Curvature: `CVR_k = −min(V(x+RW·x) − V(x) − RW·x·δ_k , V(x−RW·x) − V(x) + RW·x·δ_k)`.

## Verification (`FrtbSaSbmTest`)

An **independent** nested aggregation, written fresh in the test (not calling
`NestedAggregation`), reproduces the calculator's delta + vega + curvature for
**every** correlation scenario on a mixed long/short two-bucket book, and the
reported total is the largest of the three.

## Not in this slice

GIRR / CSR / commodity / FX risk classes; the equity repo-rate factor;
the FRTB/CRR3 COREP relieved-vs-unrelieved dual (two replays, no recompile —
mechanically the scenario DSL). All build on the same `NestedAggregation`.
