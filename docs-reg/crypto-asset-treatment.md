# Crypto-Asset Prudential Treatment — Compute Profile & NablaTensor Fit

> **Analysis note** — where the heaviest computation in the crypto-asset
> prudential regime sits and whether NablaTensor helps. Not committed scope.
> **Date:** 2026-09-02.
>
> **Verdict: Not a driver** — the rule is classification and risk-weighting, with
> no quantitative-pricing component; only crypto *trading* books pull in the FRTB
> SA machinery, and then it is just `frtb-sa.md` on a new underlying.
>
> **Calculators, not sign-off.** This note describes where the computation sits
> and what NablaTensor could compute. Model validation, parameter attestation and
> regulatory submission stay with the user.

---

## 1 · What the rule requires

The Basel prudential standard for banks' crypto-asset exposures (**SCO60**):
classify exposures into **Group 1** (tokenised traditional assets, stablecoins
meeting conditions) and **Group 2** (everything else — punitive **1250 % risk
weight**), apply an exposure cap, and give trading positions FRTB-style
treatment.

**Binding dates.** Basel implementation date **1 January 2026**. 🇪🇺 CRR3
introduces a **transitional regime** with specific risk weights, **Group 2
capped at 1 % of Tier 1**; the EBA final report on draft RTS was published
5 Aug 2025; the full Commission legislative proposal is not yet published as of
Sept 2026. 🇬🇧 the PRA has consulted; UK rules expected to track Basel.

## 2 · Where the computational load is

Almost none. The core of the rule is:

- **Classification** — assign each exposure to Group 1a / 1b / 2a / 2b against
  qualitative conditions.
- **Risk-weighting and aggregation** — apply the prescribed weight, test the
  Group 2 exposure against the Tier 1 cap.

That is reference-data and rules work — not floating-point-bound. The **only**
piece with quantitative content is where a bank holds **tokenised-asset trading
positions or crypto derivatives**, which then get **FRTB SA / IMA** sensitivities
treatment — i.e. the workload already described in `frtb-sa.md`, on a new
underlying.

**The bottleneck (such as it is) is exposure classification and data
aggregation; there is no revaluation problem unless a crypto trading book
exists.**

## 3 · Can NablaTensor help?  **Not a driver**

Barely, and only for the handful of banks with a real crypto trading book:

- For a crypto **trading** book, the delta / vega / curvature sensitivities and
  curvature repricings are the same FRTB SA path (`frtb-sa.md`), so any speed-up
  there carries over — but that is FRTB, not this rule.
- **What it does not touch:** Group classification, the stablecoin-condition
  tests, the Tier 1 cap, and the banking-book risk-weight lookup — the substance
  of SCO60 / the CRR3 transitional regime.

## 4 · If we build it

**Not on the roadmap.** Keep on the radar only: if a client's crypto trading
book becomes material, it is covered by the FRTB SA work (Phase 1) with a new
risk-factor / bucket set, and needs no dedicated module.

## References

- Basel Framework **SCO60** (prudential treatment of cryptoasset exposures) —
  `https://www.bis.org/basel_framework/`.
- EU **CRR3** (Regulation (EU) 2024/1623), crypto-asset transitional regime.
- [CRR III – prudential treatment of crypto exposures (White & Case)](https://www.whitecase.com/insight-alert/crr-iii-prudential-treatment-crypto-exposures)
- [EBA publishes draft technical standards on the prudential treatment of crypto-asset exposures under the CRR](https://www.eba.europa.eu/publications-and-media/press-releases/eba-publishes-draft-technical-standards-prudential-treatment-crypto-asset-exposures-under-capital)
- [New rules for the crypto exposures of banks (De Nederlandsche Bank)](https://www.dnb.nl/en/sector-news/supervision-2024/new-rules-for-the-crypto-exposures-of-banks/)
- Related: `docs-reg/frtb-sa.md` (the treatment a crypto trading book falls back
  to).
