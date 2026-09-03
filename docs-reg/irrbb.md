# IRRBB — Interest-Rate Risk in the Banking Book — Compute Profile & NablaTensor Fit

> **Analysis note** — where the heaviest computation in IRRBB sits and whether
> NablaTensor helps. Not committed scope. **Date:** 2026-09-02.
>
> **Verdict: Partial fit** — a shocked-curve revaluation grid with embedded
> optionality, which is AD-friendly; the fit becomes **strong only if the
> behavioural models live inside the recorded computation** rather than in a
> separate vendor ALM system.
>
> **Calculators, not sign-off.** This note describes where the computation sits
> and what NablaTensor could compute. Model validation, parameter attestation and
> regulatory submission stay with the user.

---

## 1 · What the rule requires

Measure the sensitivity of the **economic value of equity (ΔEVE)** and **net
interest income (ΔNII)** to prescribed interest-rate shock scenarios — parallel
up / down, steepener, flattener, short-rate up / down — and report against a
**Supervisory Outlier Test (SOT)** that triggers supervisory action on a large
decline. A **standardised approach** applies where a firm's internal system is
judged inadequate. CSRBB (credit-spread risk in the banking book) is measured
alongside.

**Binding dates.** EBA Guidelines (EBA/GL/2022/14) and the SOT / standardised-
approach RTS have applied since 2022–2023. Basel **recalibrated the shock
scenarios in 2024**; the EBA's second IRRBB-heatmap report (26 Jan 2026) and
EBA/GL/2026/06 (26 Jun 2026) fold them into the SREP. Expect updated shock
scenarios feeding the SOT in **2026–2027**. 🇬🇧 PRA SS31/15 / Pillar 2A is
aligned.

## 2 · Where the computational load is

ΔEVE and ΔNII are a **full repricing of the banking book under each shocked
yield curve**:

```
(all loans, deposits, funding, banking-book derivatives)
   ×  (6+ prescribed shocked curves, more with recalibrated scenarios + CSRBB)
   ×  behavioural models: prepayment, non-maturing-deposit runoff, pipeline
```

ΔNII adds a **multi-period forward simulation** of the balance sheet (new
business, reinvestment, repricing) over the projection horizon, not a single
revaluation. The book is the whole loan / deposit portfolio — far larger in row
count than a derivatives desk — and the behavioural models are themselves
non-linear functions of the shocked rates.

**The bottleneck is the banking-book cash-flow repricing (with behavioural
optionality) repeated across every shocked curve, and the multi-period roll for
ΔNII.**

## 3 · Can NablaTensor help?  **Partial fit**

Yes, with a real caveat:

- ΔEVE under shocked curves is a **revaluation grid with embedded optionality** —
  exactly the record-once / replay-many pattern, with the shocked curves as
  scenario inputs (`com.nablatensor.scenario.ScenarioRunner`).
- The **sensitivity of ΔEVE to individual curve nodes** (and to behavioural-model
  parameters) comes from **one adjoint reverse sweep**, instead of re-running the
  ALM engine per node — useful for hedging the SOT and for attribution.
- **Caveat that decides the fit:** the payoff depends on expressing the cash-flow
  projection and the behavioural models **inside the recorded computation**. If
  prepayment and deposit models sit in a separate vendor ALM platform, only the
  final revaluation step is accelerable and the fit is weak.
- **What it does not accelerate:** behavioural-model *calibration*, new-business
  assumptions, the SOT threshold logic, and data assembly from the core banking
  system.

## 4 · If we build it

**Not in the current Phase 1–5 roadmap — a candidate.** It widens the addressable
surface
beyond derivatives desks to **ALM / treasury / finance** and fits the
scenario-DSL narrative. A build would need a banking-book cash-flow engine and
behavioural-model blocks expressed on the tape, feeding shocked-curve replay and
adjoint node sensitivities. Reuses `com.nablatensor.scenario.*` and the adjoint
engine; the curve-bootstrap Jacobian in `nablatensor-quant` maps sweep results
onto regulatory curve vertices.

## References

- Basel Framework **SRP31** (IRRBB) — `https://www.bis.org/basel_framework/`.
- [EBA — Guidelines on IRRBB and CSRBB (EBA/GL/2022/14)](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/supervisory-review-and-evaluation-process-srep-0)
- [EBA — RTS on IRRBB supervisory outlier tests](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/supervisory-review-and-evaluation-process-srep-and)
- [Second implementation report on the IRRBB heatmap (Banking.Vision, Jan 2026)](https://banking.vision/en/second-implementation-report-on-irrbb-heat-map/)
- [EBA/GL/2026/06 — revised SREP and supervisory stress-testing guidelines (26 Jun 2026)](https://www.eba.europa.eu/sites/default/files/2026-06/fd5fbfa1-2efb-4122-8e91-4831469d8150/Final%20Report%20on%20revised%20SREP%20and%20supervisory%20stress%20testing%20Guidelines.pdf)
- Related: `docs-reg/stress-testing.md` (shared scenario-replay capability).
