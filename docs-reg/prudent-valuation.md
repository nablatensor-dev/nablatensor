# Prudent Valuation / Additional Valuation Adjustments — Compute Profile & NablaTensor Fit

> **Analysis note** — where the heaviest computation in prudent valuation / AVAs
> sits and whether NablaTensor helps. Not committed scope. **Date:** 2026-09-02.
>
> **Verdict: Partial fit** — a scenario-grid revaluation problem, but a smaller
> grid than FRTB curvature or stress testing, and much of the framework is
> methodology rather than compute.
>
> **Calculators, not sign-off.** This note describes where the computation sits
> and what NablaTensor could compute. Model validation, parameter attestation and
> regulatory submission stay with the user.

---

## 1 · What the rule requires

Banks must value fair-valued positions **prudently** and book **Additional
Valuation Adjustments (AVAs)** for valuation uncertainty — market-price
uncertainty, close-out costs, model risk, concentrated positions, unearned
credit spreads, future admin costs, early termination, operational risk. AVAs are
a **direct CET1 deduction**.

**Binding dates.** The EBA RTS on prudent valuation is in force. The EBA
consulted (to 16 Apr 2024) on **targeted amendments**: reduce the observed
variability of core-approach AVAs across banks, and add an **"extraordinary
circumstances"** framework. Final RTS **pending** — a live 2026–2027 item. The
PRA maintains an equivalent regime.

## 2 · Where the computational load is

Most AVA categories are aggregation of desk-level estimates. Two carry real
compute:

- **Model-risk AVA** — reprice each fair-valued position under a **range of
  plausible alternative models / model parameters**, and take a prudent point in
  the resulting distribution.
- **Market-price-uncertainty AVA** — reprice under a **range of plausible marks**
  (bid/offer, consensus dispersion) for every unobservable or thinly-observed
  input.

Both are a **scenario-grid revaluation** across the fair-valued book:

```
(fair-valued positions)  ×  (plausible parameter / mark points per input)  ×  (per-position pricing)
```

It is conceptually a small stress test confined to valuation inputs. The grid is
smaller than FRTB curvature (a handful of points per uncertain input, not a
full shocked reprice of the whole book per risk factor).

**The bottleneck is repricing the fair-valued book across the plausible-parameter
grid for the model-risk and market-price-uncertainty AVAs.**

## 3 · Can NablaTensor help?  **Partial fit**

Yes, modestly:

- The repeated repricing under perturbed parameters and marks is exactly
  **record-once / replay-many** — build the pricing kernel once, `setInput` +
  replay per grid point through `com.nablatensor.scenario.ScenarioRunner`.
- AVA sizing **shares an engine** with independent model-validation revaluation
  (`model-risk-management.md`): the same fast, deterministic, tolerance-checked
  reprice that benchmarks front-office prices also sizes the model-risk AVA.
- Sensitivities of an AVA to its underlying uncertain inputs come from one
  adjoint sweep if a bank wants to attribute or hedge the deduction.
- **What it does not accelerate:** the aggregation formula, the diversification
  / netting-benefit calculation across AVAs, the core-approach variability
  analysis, and the governance around "plausible range" selection.

## 4 · If we build it

**Not in the Phase 1–5 plan.** It is a secondary driver that pairs naturally
with the model-validation story rather than a standalone regulatory calculator.
If pursued, it reuses `com.nablatensor.scenario.*` for the grid replay and the
adjoint engine for AVA sensitivities — no new subsystem.

## References

- EU **CRR3** Article 34 and the EBA **RTS on prudent valuation**
  (Commission Delegated Regulation (EU) 2016/101).
- [EBA consults on targeted amendments to the prudent valuation framework](https://eba.europa.eu/publications-and-media/press-releases/eba-consults-targeted-amendments-prudent-valuation-framework)
- [EBA publishes final draft technical standards on prudent valuation](https://www.eba.europa.eu/eba-publishes-final-draft-technical-standards-on-prudent-valuation)
- [European Banking Authority on prudential valuations (Forvis Mazars)](https://www.forvismazars.com/uk/en/industries/financial-services/regulatory-insights/european-banking-authority-on-pva)
- Related: `docs-reg/model-risk-management.md` (shared independent-revaluation
  engine).
