# `docs-reg/` — regulation notes and implementation plans

Working documentation for the regulatory-calculator build.

Each *implementation plan* pairs **a detailed description of the regulation** with
**the concrete plan for implementing it** in `nablatensor-reg` and the shared
risk model. The *compute-profile notes* are shorter: where the heavy computation
in a regime sits and whether NablaTensor helps.

| File | Regulation | Phase |
|---|---|---|
| [`frtb-sa.md`](frtb-sa.md) | FRTB market-risk capital — standardised approach (SBM + DRC + RRAO) | 1 |
| [`nablatensor-library-usage.md`](nablatensor-library-usage.md) | NablaTensor APIs exercised by the FRTB scenario and test-only disposition | 1 |
| [`external-nablatensor-calls.md`](external-nablatensor-calls.md) | Call-level inventory of functionality outside `nablatensor-reg` | 1 |

Planned next (full Regulation & Implementation Plans): `frtb-eu-relief.md`
(Phase 4), `trading-book-boundary.md` (Phase 5). The `cva.md` and
`output-floor.md` compute-profile notes below are the starting point for the
Phase 2 / Phase 3 plans.

## Compute-profile notes (demand-side companion)

Short per-regulation notes — where the heaviest computation sits and whether
NablaTensor helps — for every quant-relevant regime other than FRTB. Each note
reads the regulation through the `demo/frtb-full-on-cuda.sh` lens: find the one
expensive stage, then ask whether *record-once / one adjoint sweep / GPU replay*
addresses it. Verdicts: **Strong** · **Partial** · **Marginal** · **Not a
driver**.

| File | Regulation | NablaTensor fit |
|---|---|---|
| [`cva.md`](cva.md) | CVA risk capital — BA-CVA / SA-CVA | **Strong** |
| [`isda-simm.md`](isda-simm.md) | ISDA SIMM / IM model validation | **Strong** |
| [`stress-testing.md`](stress-testing.md) | Supervisory stress testing | **Strong** (headline) |
| [`model-risk-management.md`](model-risk-management.md) | Model-risk management / internal-model approval | **Strong** (strategic) |
| [`output-floor.md`](output-floor.md) | The output floor | **Partial** (transitive) |
| [`irrbb.md`](irrbb.md) | IRRBB — ΔEVE / ΔNII | **Partial** |
| [`prudent-valuation.md`](prudent-valuation.md) | Prudent valuation / AVAs | **Partial** |
| [`climate-scenario-analysis.md`](climate-scenario-analysis.md) | Climate scenario analysis | **Partial** |
| [`sa-ccr.md`](sa-ccr.md) | Counterparty credit risk — SA-CCR | **Marginal** |
| [`crypto-asset-treatment.md`](crypto-asset-treatment.md) | Crypto-asset prudential treatment | **Not a driver** |

> **Calculators, not sign-off.** These documents and the code they describe
> compute the numbers the rules ask for. Parameter attestation, model validation
> and regulatory submission remain the user's. All parameter values are
> transcribed from the cited Basel / CRR3 / PRA text and must be checked against
> the reader's current rulebook.

> **Tracking note:** `docs-reg/` is currently tracked (unlike the working notes
> kept only locally). Say so if you want it ignored instead.
