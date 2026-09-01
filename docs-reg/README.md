# `docs-reg/` — regulation notes and implementation plans

Working documentation for the regulatory-calculator build (see
`docs-internal/NABLATENSOR_REG_IMPLEMENTATION_PLAN.md` for the phase overview and
`docs-internal/NABLATENSOR_REGULATORY_DRIVERS_2026.md` for the demand side).

Each file pairs **a detailed description of the regulation** with **the concrete
plan for implementing it** in `nablatensor-reg` and the shared risk model.

| File | Regulation | Phase |
|---|---|---|
| [`frtb-sa.md`](frtb-sa.md) | FRTB market-risk capital — standardised approach (SBM + DRC + RRAO) | 1 |

Planned next: `cva.md` (BA-CVA / SA-CVA, Phase 2), `output-floor.md` (Phase 3),
`frtb-eu-relief.md` (Phase 4), `trading-book-boundary.md` (Phase 5).

> **Calculators, not sign-off.** These documents and the code they describe
> compute the numbers the rules ask for. Parameter attestation, model validation
> and regulatory submission remain the user's. All parameter values are
> transcribed from the cited Basel / CRR3 / PRA text and must be checked against
> the reader's current rulebook.

> **Tracking note:** unlike `docs-internal/` (git-ignored), `docs-reg/` is
> currently tracked. Say so if you want it ignored instead.
