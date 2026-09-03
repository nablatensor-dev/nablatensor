# Supervisory Stress Testing — Compute Profile & NablaTensor Fit

> **Analysis note** — where the heaviest computation in supervisory stress
> testing sits and whether NablaTensor helps. Not committed scope.
> **Date:** 2026-09-02.
>
> **Verdict: Strong fit (headline)** — the canonical many-scenario, many-horizon
> full revaluation and the workload behind NablaTensor's scenario-DSL story;
> NablaTensor accelerates the market-risk / revaluation slice, not the credit /
> PPNR models.
>
> **Calculators, not sign-off.** This note describes where the computation sits
> and what NablaTensor could compute. Model validation, parameter attestation and
> regulatory submission stay with the user.

---

## 1 · What the rule requires

Regulator-run exercises that project capital ratios under severe-but-plausible
macro scenarios; results feed **capital buffers** and distribution constraints.

| Programme | Cadence / date | Notes |
|---|---|---|
| 🇪🇺 EBA / ECB EU-wide stress test | biennial; **2027** exercise | 63 banks (~75 % of the EU sector), reduced data templates, **climate risk integrated** for the first time |
| 🇺🇸 Fed DFAST / CCAR / Stress Capital Buffer | annual; new framework for the **2027 cycle** | two-year averaging of the max projected CET1 decline; **public notice-and-comment on the supervisory models** |
| 🇬🇧 BoE / PRA | concurrent + desk-based; flexible cadence | scenario analysis rounds (e.g. April 2026) |

## 2 · Where the computational load is

Stress testing is the **many-scenario full revaluation** workload:

```
(every position, banking + trading book)
   ×  (multiple macro scenarios: baseline, adverse, severely adverse, climate)
   ×  (multiple forward horizons / quarters, typically 9–13 projection points)
   ×  (full revaluation + loss projection at each node)
```

This is the **heaviest total-FLOP workload of any regime covered in these
notes**. The Fed transparency proposal adds a second copy: banks build **internal replicas of the
supervisory models** and run them alongside their own.

**The bottleneck is the full-portfolio revaluation replicated across the
scenario × horizon grid — for market risk, CCR shocks and NII paths.**

## 3 · Can NablaTensor help?  **Strong fit (headline)**

Yes — this is the use case behind the scenario-DSL narrative and the
"27 h → 11 min"-style headline:

- **Record one valuation, replay the grid on the GPU.** The scenario × horizon
  grid is precisely `com.nablatensor.scenario.ScenarioRunner` /
  `ScenarioSet` replay, no recompile between scenarios.
- **Sensitivities to the scenario drivers** (rates, spreads, equity, FX paths)
  from one adjoint sweep — for attribution, for building the internal replica of
  the supervisory model, and for management overlays.
- Common random numbers across scenarios keep the projections comparable, as
  `demo/frtb-full-on-cuda.sh` does for the curvature shocks.
- **What it does not accelerate:** the credit-loss models (PD / LGD / IFRS 9
  staging), pre-provision net revenue (PPNR) projection, the macro-scenario
  generation, and balance-sheet-strategy assumptions — these dominate the
  banking-book side and sit outside a pricing engine.

## 4 · If we build it

**Not a standalone phase in the current 1–5 roadmap** — it is a horizontal
capability that the FRTB, CVA and IRRBB work all feed, and the lowest-priority
coverage item. It reuses `com.nablatensor.scenario.*` and the adjoint engine
directly; the deliverable is a scenario / horizon DSL and a results grid, not a
new regulatory calculator.

## References

- [2025 EU-wide stress test — results (EBA)](https://www.eba.europa.eu/publications-and-media/publications/2025-eu-wide-stress-test-results)
- [EBA launches early consultation on a simplified EU-wide stress test, with climate risk integration](https://www.eba.europa.eu/publications-and-media/press-releases/eba-launches-early-consultation-simplified-eu-wide-stress-test-climate-risk-integration)
- [Modifications to the capital plan rule and stress capital buffer requirement (Federal Register, 22 Apr 2025)](https://www.federalregister.gov/documents/2025/04/22/2025-06863/modifications-to-the-capital-plan-rule-and-stress-capital-buffer-requirement)
- [Enhanced transparency and public accountability of the supervisory stress test models and scenarios (Federal Register, 18 Nov 2025)](https://www.federalregister.gov/documents/2025/11/18/2025-20211/enhanced-transparency-and-public-accountability-of-the-supervisory-stress-test-models-and-scenarios)
- [The 2026 Federal Reserve stress test results: a framework in transition (Bank Policy Institute)](https://bpi.com/the-2026-federal-reserve-stress-test-results-a-framework-in-transition/)
- [Dodd-Frank Act Stress Tests 2026 (Federal Reserve)](https://www.federalreserve.gov/supervisionreg/dfa-stress-tests-2026.htm)
- Related: `docs-reg/irrbb.md`, `docs-reg/climate-scenario-analysis.md` (same
  scenario-replay capability).
