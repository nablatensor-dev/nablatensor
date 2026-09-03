# Counterparty Credit Risk — SA-CCR — Compute Profile & NablaTensor Fit

> **Analysis note** — where the heaviest computation in SA-CCR sits and whether
> NablaTensor helps. Not committed scope. **Date:** 2026-09-02.
>
> **Verdict: Marginal** — the base calculation is a closed-form formula and is
> not compute-bound; there is no deadline pressure.
>
> **Calculators, not sign-off.** This note describes where the computation sits
> and what NablaTensor could compute. Model validation, parameter attestation and
> regulatory submission stay with the user.

---

## 1 · What the rule requires

The standardised approach for measuring counterparty credit-risk exposure at
default on derivatives:

```
EAD  =  α · ( RC + PFE )          α = 1.4
PFE  =  multiplier · Σ_hedging-set  add-on
add-on(trade)  uses a supervisory delta  δ  (a Black-Scholes-style formula for
               option-like trades) and prescribed supervisory factors
```

**Binding dates.** **Already in force** — EU (CRR2, 2021), US (2022), UK. This is
a *steady-state* obligation, not a new deadline. Industry (AFME, EBF, ISDA)
continues to press for recalibration of `α = 1.4` and the add-ons; nothing
adopted as of Sept 2026.

## 2 · Where the computational load is

SA-CCR is a **deterministic analytic formula**. Per netting set it evaluates in
microseconds: replacement cost, per-trade supervisory deltas and maturity
factors, hedging-set aggregation, one multiplier. There is no simulation and no
iterative solve.

The only real cost is **scale and repetition**:

- Running the formula across **millions of trades / thousands of counterparties**
  on every reporting date.
- Re-running it inside **what-if / optimisation loops** — collateral terms,
  netting-set membership, trade compression — where the same closed form is
  evaluated thousands of times over candidate portfolios.

**The bottleneck is throughput over a very large trade population and inside
optimisation loops — not the cost of any single evaluation.**

## 3 · Can NablaTensor help?  **Marginal**

Mostly no.

- The supervisory deltas are **simple closed forms** that do not need adjoint
  AD; the hedging-set aggregation is data-plumbing, not floating-point-bound.
- A genuine fit exists only where **EAD sensitivities** are wanted — SA-CCR EAD
  feeds the leverage ratio, large-exposures limits and the output floor, so a
  desk may want `dEAD/d(market factor)` or `dEAD/d(collateral)` — or where
  SA-CCR sits **inside an optimiser** and a gradient accelerates the search. In
  those cases one adjoint sweep of the (short, analytic) EAD computation gives
  the full gradient in one pass.
- **What it does not accelerate:** the base regulatory number, trade
  decomposition, and reference-data / netting-set assembly.

## 4 · If we build it

**Not currently a build phase.** SA-CCR appears in the roadmap only as a
downstream input to the output floor (Phase 3), which takes EAD as given. If a
client optimisation or leverage-ratio use case emerges, the natural home is a
thin analytic module differentiated by the existing adjoint engine — no new
subsystem — feeding `com.nablatensor.risk` position arithmetic.

## References

- Basel Framework **CRE52** (SA-CCR) — `https://www.bis.org/basel_framework/`.
- EU **CRR2/CRR3**, Part Three Title II Chapter 6 Section 3.
- [The Standardised Approach for Counterparty Credit Risk — design and calibration (AFME)](https://www.afme.eu/publications/position-papers/the-standardised-approach-for-counterparty-credit-risk-design-and-calibration/)
- [Recalibration of SA-CCR to mitigate increased hedging costs for end-users (AFME)](https://www.afme.eu/publications/position-papers/recalibration-sa-ccr-to-mitigate-increased-hedging-costs-for-end-users/)
- [Review of the framework for SA-CCR — EBF position](https://www.ebf.eu/regulation-supervision/review-of-the-framework-for-the-standardised-approach-for-counterparty-credit-risk-sa-ccr-ebf-position/)
- Related: `docs-reg/output-floor.md` (SA-CCR EAD as a downstream input).
