# ISDA SIMM (equity)

*Keywords: ISDA SIMM java, SIMM initial margin equity, SIMM concentration risk factor, SIMM delta vega margin open source*

SIMM shares the two-level aggregation with [FRTB SA-SBM](frtb-sa-sbm.md) and adds
the **concentration risk factor** `CR_k`. The sensitivity vector is, again, one
adjoint sweep of the book.

> **Parameter values are illustrative, not the ISDA calibration.** SIMM is
> recalibrated annually and its risk weights / correlations / thresholds are
> version-specific and licence-restricted. `SimmEquityParameters` ships
> structurally-correct placeholders; replace them with the current ISDA-published
> set for any real use. `IsdaSimm` is fully parameter-driven, so only that class
> changes.

## Coverage

- **Risk class:** equity. **Margins:** delta + vega (with `HVR` scaling).
- **Concentration:** `CR_k = max(1, √(|s_k| / T_b))` scales every weighted
  sensitivity and enters the within-bucket cross terms as
  `f_kl = min(CR_k, CR_l) / max(CR_k, CR_l)`. The across-bucket `g_bc`
  correction is treated as 1 in this slice.

```java
Sensitivities book = ...;                 // equity delta + vega RiskFactors
IsdaSimm.Result r = IsdaSimm.equity(book);
r.deltaMargin(); r.vegaMargin(); r.total();
```

## Aggregation

```
CR_k  = max(1, √(|s_k| / T_bucket))
WS_k  = RW_bucket · s_k · CR_k
K_b   = √(Σ WS_k² + Σ_{k≠l} ρ_kl · f_kl · WS_k WS_l)
S_b   = max(min(Σ WS_k, K_b), −K_b)
margin= √(Σ K_b² + Σ_{b≠c} γ_bc S_b S_c)
```

## Verification (`IsdaSimmTest`)

A hand-worked delta margin — `CR_k`, `f_kl`, the within- and across-bucket
sums — computed fresh in the test reproduces `IsdaSimm.equity(...)` for a
three-name book sized to trip the concentration threshold.

## Not in this slice

Other SIMM risk classes; the curvature (scaled-vega) margin; the `g_bc`
across-bucket concentration correction; the product-class and risk-class
combination formula. All build on the same `NestedAggregation`.
