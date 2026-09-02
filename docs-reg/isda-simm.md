# ISDA SIMM & IM Model Validation — Compute Profile & NablaTensor Fit

> **Analysis note** — companion to
> `docs-internal/NABLATENSOR_REGULATORY_DRIVERS_2026.md` §8. Not committed scope.
> **Date:** 2026-09-02.
>
> **Verdict: Strong fit** — the same sensitivity vectors as FRTB SA, plus a
> historical-replay backtest; a twice-a-year, every-year obligation for both
> sell-side and buy-side.
>
> **Calculators, not sign-off.** This note describes where the computation sits
> and what NablaTensor could compute. Model validation, parameter attestation and
> regulatory submission stay with the user.

---

## 1 · What the rule requires

For **non-cleared derivatives** above the €50 m / $50 m threshold, counterparties
exchange **initial margin**. Most of the market uses **ISDA SIMM** — a
**sensitivities-based** model: delta, vega and curvature by risk class, with
concentration thresholds, aggregated with ISDA-calibrated risk weights and
correlations.

**Cadence and dates.**

- All in-scope firms are live (Uncleared Margin Rules Phase 6, Sep 2022).
- SIMM moved to **semiannual recalibration**: v2.8+2506 (31 Oct 2025),
  v2.8+2512 (12 Jun 2026); each version adopted on a common go-live date.
- Firms run **annual industry backtesting and benchmarking**, submitting
  sensitivity data to ISDA.
- 🇪🇺 Under **EMIR 3.0** (in force 24 Dec 2024) every firm exchanging IM must
  submit its **IM model for validation** to its NCA; the EBA is building the
  validation RTS / central function.

## 2 · Where the computational load is

Two distinct workloads:

1. **Sensitivity generation.** The full **delta / vega / curvature vector** for
   every trade in the non-cleared book — the *same* sensitivities as FRTB SA
   (different risk weights). Computed **daily** for margin calls and regenerated
   in full at **each recalibration**. By bump-and-revalue this is `O(#risk
   factors)` repricings per trade.
2. **Annual backtesting.** Recompute SIMM IM **and** the realised portfolio P&L
   over **long historical windows** (years of daily data) for every counterparty
   relationship, then compare against realised moves:

   ```
   (counterparties)  ×  (thousands of historical dates)  ×  (full portfolio reval + SIMM recompute)
   ```

**The bottleneck is (1) the daily / per-recalibration full sensitivity set and
(2) the historical-window portfolio revaluation for the annual backtest.**

## 3 · Can NablaTensor help?  **Strong fit**

Yes, on both workloads:

- **Sensitivities in one sweep.** Record the portfolio valuation once; one
  adjoint reverse sweep yields the whole delta / vega / curvature vector, instead
  of bump-and-revalue per risk factor. This is the exact "one sweep vs N
  revaluations" comparison in `demo/adjoint-vs-bump.sh`.
- **Backtest as replay-many.** One recorded valuation replayed across thousands
  of historical market states on the GPU is a textbook NablaTensor workload
  (`demo/greeks-on-gpu.sh` spot-ladder / crash-scenario pattern, scaled up).
- The SIMM aggregation itself — buckets, concentration factor `CR_k`,
  within/across correlations — is already implemented for equity in
  `com.nablatensor.reg.simm` on the shared `NestedAggregation` engine.
- **What it does not accelerate:** ISDA's own recalibration of risk weights from
  pooled data, the version-adoption coordination, dispute reconciliation, and
  the regulatory validation filing.

## 4 · If we build it

Extends the existing **`com.nablatensor.reg.simm`** slice (equity delta + vega +
concentration) to all risk classes, curvature and the `g_bc` cross-bucket term —
listed as coverage priority 3 in the drivers doc (`§14`). The recalibration
cadence and the annual backtest are the recurring commercial hook; a backtest
harness over historical market snapshots would reuse `com.nablatensor.scenario.*`.

## References

- [ISDA publishes ISDA SIMM Methodology v2.8+2506 (31 Oct 2025)](https://www.isda.org/2025/10/31/isda-publishes-isda-simm-methodology-version-2-8-2506/)
- [ISDA publishes ISDA SIMM Methodology v2.8+2512 (12 Jun 2026)](https://www.isda.org/2026/06/12/isda-publishes-isda-simm-methodology-version-2-8-2512/)
- [ISDA SIMM — solutions info hub](https://www.isda.org/isda-solutions-infohub/isda-simm/)
- [Initial margin for non-cleared derivatives: the end of the journey? (BNP Paribas)](https://securities.cib.bnpparibas/initial-margin-for-non-cleared-derivatives-the-end-of-the-journey/)
- EU **EMIR 3.0** (Regulation (EU) 2024/2987), IM model-validation provisions.
- Internal: `docs-internal/NABLATENSOR_REGULATORY_DRIVERS_2026.md` §8;
  existing `com.nablatensor.reg.simm`.
