# CVA Risk Capital (BA-CVA / SA-CVA) — Compute Profile & NablaTensor Fit

> **Analysis note** — companion to
> `docs-internal/NABLATENSOR_REGULATORY_DRIVERS_2026.md` §3. Not committed scope.
> **Date:** 2026-09-02.
>
> **Verdict: Strong fit** — the widest adjoint-AD margin in the whole drivers
> document.
>
> **Calculators, not sign-off.** This note describes where the computation sits
> and what NablaTensor could compute — the numbers the rules ask for. Model
> validation, parameter attestation and regulatory submission stay with the user.

---

## 1 · What the rule requires

Capital for the risk of mark-to-market losses on the credit valuation adjustment
of a derivative portfolio. Two approaches:

| Approach | Shape | Compute weight |
|---|---|---|
| **BA-CVA** (basic) | closed-form formula on hedged / unhedged notional-style inputs | negligible |
| **SA-CVA** (standardised) | a **sensitivities-based** charge — CVA **delta and vega** to prescribed credit-spread, rates, FX, equity and commodity risk factors, aggregated FRTB-style; needs supervisory approval and a CVA desk | heavy — see §2 |

**Binding dates.** With FRTB in each jurisdiction: 🇪🇺 **1 Jan 2027** (inside the
same CRR3 targeted-relief delegated act), 🇺🇸 **2027** phase-in, 🇬🇧 **1 Jan
2027**. UK Basel 3.1 **eliminates all CVA internal models** and replaces them
with three risk-sensitive standardised methods, so every UK bank with material
CVA needs a sensitivities engine — not a Monte-Carlo IMM-CVA model — for capital.

## 2 · Where the computational load is

The CVA of a netting set is itself an expectation, in practice a **Monte-Carlo
simulation of counterparty exposure**:

```
CVA(netting set)  ≈  E[ LGD · Σ_t  discount_t · EE_t · dPD_t ]
```

where each exposure path requires a **full revaluation of every trade in the
netting set at every simulation time step**. The cost shape is:

```
N_paths  ×  N_time_steps  ×  (trades in netting set)  ×  (per-trade pricing)
```

The SA-CVA capital charge then needs CVA **delta and vega to every prescribed
risk factor** — dozens to a few hundred credit-spread, rates, FX, equity and
commodity factors. Computed by bump-and-revalue, that is the entire exposure
simulation **re-run once per risk factor**.

- Monte-Carlo exposure simulation — the dominant cost, and it is re-paid per
  bumped factor.
- FRTB-style nested aggregation of the weighted sensitivities — cheap arithmetic.
- BA-CVA — a closed form; nothing to accelerate.

**The bottleneck is the per-risk-factor re-simulation of the netting-set
exposure paths for the SA-CVA sensitivity vector.**

## 3 · Can NablaTensor help?  **Strong fit**

Yes — this is where bump-and-revalue is most expensive and adjoint AD wins by the
widest margin in the drivers document.

- **Record the netting-set CVA valuation once.** One adjoint reverse sweep
  produces `dCVA/d(risk factor)` for **all** prescribed factors together, instead
  of hundreds of bumped re-simulations — the cost becomes `O(1)` sweeps rather
  than `O(#risk factors)` simulations.
- **Replay the exposure paths on the GPU.** The path simulation is the same
  record-once / replay-many workload the `demo/greeks-on-gpu.sh` and
  `demo/frtb-full-on-cuda.sh` scripts already show for option books.
- The weighted-sensitivity aggregation reuses `com.nablatensor.risk.NestedAggregation`
  — the same engine FRTB SA-SBM uses, with CVA risk weights and correlations.

**What it does not accelerate:** BA-CVA (closed form), the eligibility /
approval / desk-governance work, and hedge-instrument mapping.

## 4 · If we build it

Internal implementation plan **Phase 2** (CVA risk capital) — the largest new
sub-system on the roadmap: it needs a counterparty-exposure engine feeding the
existing aggregation layer. Extends `com.nablatensor.risk.NestedAggregation`
(FRTB-style aggregation, already generic) and the adjoint engine in
`nablatensor-core`; a fuller *Regulation & Implementation Plan* in the style of
`frtb-sa.md` would supersede this note once Phase 2 is scoped.

Headline comparison artefact: "SA-CVA sensitivities for a real netting set — one
adjoint sweep vs N re-simulations."

## References

- Basel Framework **MAR50** (CVA risk) and **CRE** counterparty-credit chapters —
  `https://www.bis.org/basel_framework/`.
- EU **CRR3** (Regulation (EU) 2024/1623), CVA chapter; targeted-relief delegated
  act C(2026) 3647 final.
- UK **PRA PS1/26**, *Implementation of Basel 3.1: Final rules* (20 Jan 2026) —
  standardised CVA, no internal models.
- [A first view on the new CVA risk capital charge (Quantifi)](https://www.quantifisolutions.com/a-first-view-on-the-new-cva-risk-capital-charge-1/)
- [FRTB may bite harder for Europe's CVA modellers (Risk.net)](https://www.risk.net/our-take/7961311/frtb-may-bite-harder-for-europes-cva-modellers)
- [EBA — market, counterparty and CVA risk](https://www.eba.europa.eu/regulation-and-policy/market-counterparty-and-cva-risk)
- Internal: `docs-internal/NABLATENSOR_REGULATORY_DRIVERS_2026.md` §3,
  `docs-internal/NABLATENSOR_REG_IMPLEMENTATION_PLAN.md` Phase 2.
