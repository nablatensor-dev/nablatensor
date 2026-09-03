# The Output Floor — Compute Profile & NablaTensor Fit

> **Analysis note** — where the heaviest computation the output floor forces sits
> and whether NablaTensor helps. Not committed scope. **Date:** 2026-09-02.
>
> **Verdict: Partial fit (transitive)** — the floor is not a new calculation; it
> makes the standardised approaches into permanent production compute, so it
> inherits the FRTB SA and SA-CVA speed-ups and nothing more.
>
> **Calculators, not sign-off.** This note describes where the computation sits
> and what NablaTensor could compute. Model validation, parameter attestation and
> regulatory submission stay with the user.

---

## 1 · What the rule requires

A floor on total risk-weighted assets computed with internal models, set at
**72.5 % of the RWA the standardised approaches would produce**. Practical
effect: an IMA / IRB bank must **also run the full standardised approach every
reporting cycle** — including **FRTB SA** and **SA-CVA** — and hold the higher of
the two numbers.

**Binding dates.** 🇪🇺 phase-in **50 % (2025) → 55 % (2026) → 60 % (2027) →
72.5 % on 1 Jan 2030** (with transitional softeners to 2032–33). 🇬🇧 phased **1
Jan 2027 → 1 Jan 2030**. 🇺🇸 the March 2026 re-proposal **removes** the output
floor entirely.

## 2 · Where the computational load is

The floor adds **no new formula**. What it adds is an obligation: the entire
standardised RWA stack must run **in parallel** with the internal models, as a
monthly / quarterly production number rather than a fallback, indefinitely, at an
escalating multiplier.

For the market-risk slice that means, every cycle:

- **FRTB SA** — the sensitivities-based method plus the curvature **shocked
  repricings** (the one expensive stage, per `frtb-sa.md` §I.3 and
  `demo/frtb-full-on-cuda.sh`).
- **SA-CVA** — the netting-set exposure re-simulation for the CVA sensitivity
  vector (per `cva.md` §2).
- Standardised credit-risk RWA, SA-CCR EAD, standardised operational risk — the
  bulk of the number, but not a market-risk-analytics problem.

**The bottleneck is whatever FRTB SA and SA-CVA cost — now paid on every
reporting cycle, forever, instead of once at migration.**

## 3 · Can NablaTensor help?  **Partial fit (transitive)**

Yes, but only through its FRTB SA and SA-CVA contributions:

- Every speed-up in `frtb-sa.md` (curvature repricings on the GPU, adjoint
  delta / vega) and `cva.md` (one-sweep SA-CVA sensitivities) applies here
  unchanged.
- The floor is what makes those speed-ups **structural rather than one-off**: a
  faster standardised engine is not a migration convenience, it is a permanent
  reduction in a recurring production workload for every EU and UK internal-model
  bank through 2030.
- **What it does not accelerate:** standardised credit / operational RWA, the
  floor arithmetic itself (a single `max`), COREP assembly, and the
  reconciliation between the internal and standardised runs.

## 4 · If we build it

**Phase 3** of the regulatory-calculator build — and deliberately narrow:
NablaTensor's slice is the **MR-SA and SA-CVA legs only**, consuming the totals
produced by Phases 1 and 2 (`docs-reg/frtb-sa.md`, `docs-reg/cva.md`). Standardised credit and operational RWA are out of scope (not a
quantitative-pricing problem). The deliverable is the parallel-run plumbing and
the `max` against the internal-model number, plus the relieved / unrelieved dual
run already sketched in `frtb-sa.md` §I.6.

## References

- Basel Framework **RBC20.10–20.14** (output floor) — `https://www.bis.org/basel_framework/`.
- EU **CRR3** (Regulation (EU) 2024/1623), Article 92(3) and transitional Article 465.
- UK **PRA PS1/26**, *Implementation of Basel 3.1: Final rules* (20 Jan 2026).
- [Basel III timeline: output floor 2025–2030 & CRR III deadlines (ADVISORI)](https://www.advisori.de/services/regulatory-compliance-management/basel-iii/basel-iii-implementation-timeline)
- [CRR III — transitional provisions (Banking.Vision)](https://banking.vision/en/crr-iii-transitional-provisions)
- [Credit risk & output floor (Ashurst)](https://www.ashurst.com/en/insights/credit-risk-and-output-floor/)
- [EBA roadmap on the EU banking package — the output floor](https://probability.nl/wp-content/uploads/2024/10/EBA-roadmap-on-EU-Banking-package-Output-Floor-PP-1.pdf)
- Related: `docs-reg/frtb-sa.md`, `docs-reg/cva.md` (the two standardised legs
  this floor makes permanent).
