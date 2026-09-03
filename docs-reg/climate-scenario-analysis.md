# Climate Scenario Analysis — Compute Profile & NablaTensor Fit

> **Analysis note** — where the heaviest computation in climate scenario analysis
> sits and whether NablaTensor helps. Not committed scope. **Date:** 2026-09-02.
>
> **Verdict: Partial fit** — mechanically the same replay-across-a-grid problem
> as stress testing, but a softening, EU/UK-only driver with still-maturing
> methodology; a supporting example that rides on the stress-testing capability.
>
> **Calculators, not sign-off.** This note describes where the computation sits
> and what NablaTensor could compute. Model validation, parameter attestation and
> regulatory submission stay with the user.

---

## 1 · What the rule requires

Forward-looking analysis of how **transition** and **physical** climate risk flow
through credit, market and operational losses over long horizons.

- 🇪🇺 **ECB** — the one-off Fit-for-55 system-wide climate scenario analysis
  (2024), plus recurring incorporation of climate-policy effects into the ECB
  macro projections (reassessed for 2026–2028); SSM climate stress work
  continues.
- 🇬🇧 **Bank of England** — contributed to the **NGFS short-term climate
  scenarios** and publishes its own scenario analysis; the earlier CBES informs
  supervisory expectations.
- 🇺🇸 **Federal Reserve** — stepped back from dedicated climate scenario work and
  left the NGFS in early 2025; **not currently an active US driver**.

## 2 · Where the computational load is

A **scenario-grid revaluation** along long-horizon pathways:

```
(banking- + trading-book positions)
   ×  (NGFS-style pathways: orderly / disorderly / hot-house, + carbon price, GDP, rates)
   ×  (annual time steps out to 2050)
   ×  (revaluation + counterparty-level loss projection)
```

The scenario grid is **larger than a conventional stress test** — decades of
annual steps across several correlated macro pathways — but each revaluation is
the same kind of operation, and the methodology (sector transition paths,
counterparty-level carbon sensitivity) is still evolving.

**The bottleneck is the position-level revaluation replicated across the pathway
× long-horizon grid.**

## 3 · Can NablaTensor help?  **Partial fit**

Yes in principle, but this is not a lead:

- The pathway × horizon grid is the **same record-once / replay-many** structure
  as `stress-testing.md`; if that capability exists, climate scenario analysis
  reuses it unchanged, with the NGFS macro variables as scenario inputs.
- Sensitivities of projected losses to the pathway drivers (carbon price, rates)
  come from one adjoint sweep.
- **Why it stays "partial":** the driver is **softening**, EU/UK-only, and the
  methodology is immature, so there is little procurement pull specific to
  climate beyond what stress testing already justifies.
- **What it does not accelerate:** the climate-economic scenario construction,
  sector transition modelling, physical-hazard mapping, and counterparty-level
  emissions data.

## 4 · If we build it

**Not on the roadmap as a distinct item.** Treat as a supporting example layered
on the stress-testing scenario harness (`stress-testing.md` §4): swap the macro
scenario set for NGFS pathways, extend the horizon, reuse
`com.nablatensor.scenario.*` and the adjoint engine.

## References

- [ECB Fit-for-55 climate scenario analysis — report](https://www.ecb.europa.eu/pub/pdf/other/ecb.report_fit-for-55_stress_test_exercise~7fec18f3a8.en.pdf)
- [ECB — climate change and monetary policy (speech, 5 May 2026)](https://www.ecb.europa.eu/press/key/date/2026/html/ecb.sp260505_1~2e47b4c747.en.html)
- [Bank of England — international engagement and initiatives (NGFS short-term scenarios)](https://www.bankofengland.co.uk/climate-change/international-engagement-and-initiatives)
- [Comparing the ECB SSM climate stress test and the Bank of England's CBES (Deloitte)](https://www.deloitte.com/uk/en/services/audit-assurance/blogs/comparing-the-ecb-ssm-climate-change-stress-and-the-bank-of-englands-cbes.html)
- Related: `docs-reg/stress-testing.md` (the scenario-replay capability this
  rides on).
