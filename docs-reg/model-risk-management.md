# Model-Risk Management & Internal-Model Approval — Compute Profile & NablaTensor Fit

> **Analysis note** — companion to
> `docs-internal/NABLATENSOR_REGULATORY_DRIVERS_2026.md` §11. Not committed scope.
> **Date:** 2026-09-02.
>
> **Verdict: Strong fit (strategic)** — the strongest strategic fit in the
> drivers document; it needs no new regulatory calculator, only NablaTensor's
> existing independent-replay / oracle machinery, positioned as the challenger /
> validation engine and **never the book of record**.
>
> **Calculators, not sign-off.** This note describes where the computation sits
> and what NablaTensor could compute. Model validation, parameter attestation and
> regulatory submission stay with the user.

---

## 1 · What the rule requires

Supervisory expectations that firms manage **model risk as a discipline**:
inventory, tiering, **independent validation**, benchmarking, challenger models,
ongoing monitoring — and, for regulatory-capital models, formal supervisory
**approval and change governance**.

| Jurisdiction | Instrument | Status |
|---|---|---|
| 🇬🇧 | **PRA SS1/23** (PS6/23) — five principles, incl. **Principle 4: independent validation**; covers AI/ML | **effective 17 May 2024**; ongoing supervisory review |
| 🇪🇺 | **ECB Guide to Internal Models (EGIM)** — CRR3 / FRTB chapters | **June 2026 update** (own-debt identification, ES-percentile correction) |
| 🇺🇸 | **SR 11-7** | baseline; underpins every IMA / IMM approval |

## 2 · Where the computational load is

The compute is not a regulatory *formula* — it is the obligation to stand up a
**second, independent revaluation**:

- An **independent Greeks / pricing engine** — ideally a different
  implementation and language from the front office — that reprices the same
  book and reconciles to a tolerance.
- **Challenger models** run in parallel with the production model.
- **Benchmarking** and error metrics computed repeatedly, for **every FRTB / CVA
  / CCR model going live in 2027** and on every material model change.

```
(every capital / pricing model)  ×  (independent reprice of its book)  ×  (each validation cycle + each change)
```

**The bottleneck is running a full, independent revaluation of every in-scope
book, repeatedly, on a separate code path from the front office.**

## 3 · Can NablaTensor help?  **Strong fit (strategic)**

Yes — this maps one-to-one onto what NablaTensor already is:

- **Deterministic replay / CPU-oracle validation.** A recorded valuation replays
  bit-reproducibly and can be cross-checked against a scalar CPU oracle to a
  tolerance — exactly `demo/one-tape-every-backend.sh` ("every row cross-checked
  against the scalar `cpu` oracle"). That is a concrete instance of SS1/23
  Principle 4 and the EGIM internal-validation expectations.
- **Independent by construction.** A `java.lang.foreign` FFI engine with
  runtime-compiled kernels is a genuinely different implementation path from a
  typical C++ / Python front-office library.
- **Challenger Greeks in one sweep** for benchmarking against the production
  model's sensitivities.
- **Positioning:** the challenger / validation engine, never the book of record —
  consistent with the "calculators, not sign-off" disclaimer carried through
  `docs-reg/*` and the `*Parameters` classes.
- **What it does not accelerate:** the governance, documentation, model
  inventory / tiering, and the independent-review sign-off itself.

## 4 · If we build it

**Nothing new to build as a regulatory calculator.** The capability already
exists: the record / replay engine, the bit-repro check and the CPU oracle in
`nablatensor-validate` (`BitReproTest`, `BumpCrossCheck`). The work is packaging
and documenting it as a validation workflow — a "challenger / oracle" guide in
`docs-reg/` — and wiring the tolerance-checked reconciliation report. This
supports every other phase rather than sitting in the Phase 1–5 sequence.

## References

- [SS1/23 – Model risk management principles for banks (Bank of England)](https://www.bankofengland.co.uk/prudential-regulation/publication/2023/may/model-risk-management-principles-for-banks-ss)
- [PS6/23 – Model risk management principles for banks (Bank of England)](https://www.bankofengland.co.uk/prudential-regulation/publication/2023/may/model-risk-management-principles-for-banks)
- [ECB Guide to Internal Models — June 2026 (PDF)](https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.supervisory_guide_egim_202606.en.pdf)
- [ECB Guide to Internal Models: what changed in the 2026 update (Better Regulation)](https://betterregulation.com/insights/posts/ecb-guide-to-internal-models-2026-update.html)
- [SR 11-7: Guidance on Model Risk Management (Federal Reserve)](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
- Internal: `docs-internal/NABLATENSOR_REGULATORY_DRIVERS_2026.md` §11;
  `nablatensor-validate`, `demo/one-tape-every-backend.sh`.
