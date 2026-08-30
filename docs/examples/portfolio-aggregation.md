# Portfolio aggregation (Seam 7)

*Keywords: portfolio greeks aggregation, netting set sensitivities, per-trade adjoint aggregate, risk factor bucketing java*

Each trade is recorded and risked on its own tape. The portfolio, netting-set
and bucket views are **plain addition** of the resulting sensitivity vectors —
the aggregation layer (`nablatensor-risk`) never touches a kernel.

## Types

| type | role |
|---|---|
| `RiskFactor(riskClass, measure, bucket, name, tenor)` | the key a sensitivity is weighted and bucketed by |
| `Sensitivities` | immutable `RiskFactor → value`; `plus`, `scaled`, `filter`, `ofClass`, `ofMeasure`, `inBucket` |
| `Portfolio` / `Portfolio.Trade` | a book of trades; `aggregate()`, `byNettingSet()` |
| `NestedAggregation` | the two-level `√(Σ K_b² + Σ γ_bc S_b S_c)` engine — see below |
| `CorrelationScenario` | FRTB's `LOW` / `MEDIUM` / `HIGH` transforms |
| `TimeProfile` | exposure on a time grid (EPE/ENE/peak/weighted integral) — the XVA hook |

```java
Portfolio book = new Portfolio(trades.stream()
    .map(t -> Portfolio.trade(t.id(), t.nettingSet(), t.adjointSensitivities()))
    .toList());

Sensitivities delta = book.aggregate().ofClass(RiskClass.EQUITY).ofMeasure(RiskMeasure.DELTA);
Map<String, Sensitivities> perNettingSet = book.byNettingSet();
```

## Verification (`PortfolioAggregationTest`)

The Phase-2 definition of done: **per-trade adjoint sensitivities, aggregated to
the book, reconcile to a full one-factor-at-a-time bump grid** on the whole
book — for a mixed long/short book across two equity names in two SBM buckets,
to `1%`. The netting-set split (`NS_A + NS_B`) reconstructs the book exactly.

## The nested aggregation

`NestedAggregation` is the shared spine of FRTB SA-SBM and ISDA SIMM:

```
WS_k  = RW_k · s_k                        (curvature feeds CVR_k directly)
K_b   = √(max(0, Σ D(WS_k) + Σ_{k≠l} R(ρ_kl) WS_k WS_l ψ_kl))
S_b   = clamp(Σ WS_k, −K_b, K_b)
total = √(max(0, Σ K_b² + Σ_{b≠c} R(γ_bc) S_b S_c ψ_bc))
```

delta/vega: `D(w)=w²`, `R(c)=c`, `ψ=1`; curvature: `D(w)=max(w,0)²`, `R(c)=c²`,
`ψ(a,b)=0` iff both negative. `withConcentration(CR)` adds the SIMM
`CR_k`-scaling and `f_kl = min/max` within-bucket correction.
`NestedAggregationTest` reconciles all of this to hand arithmetic.
