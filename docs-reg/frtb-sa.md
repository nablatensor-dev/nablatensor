# FRTB Standardised Approach — Regulation & Implementation Plan

> **Phase 1** of the regulatory-calculator build. Companion to
> `docs-internal/NABLATENSOR_REG_IMPLEMENTATION_PLAN.md`.
> **Status:** Part I is the regulation; Part II is largely **implemented** in
> `nablatensor-reg` — SBM for all seven risk classes, DRC, RRAO and the `FrtbSa`
> assembler, with independent-oracle tests. Still open: the prescribed-bump /
> adjoint **sensitivity extraction** bridge (§II.8), calibrated CSR
> securitisation / CTP tables, and the runnable multi-class example. **Date:**
> 2026-09-01.
>
> **Calculators, not sign-off.** This describes what the code will compute — the
> numbers MAR21/22/23 and CRR3 ask for. Model validation, parameter attestation
> and regulatory submission stay with the user. Every numeric table below is
> *indicative*, transcribed from the cited text; the code keeps the authoritative
> copy in a `*Parameters` class with a "verify against your regulator's current
> rulebook" warning.

---

## Contents

**Part I — The regulation**
- [I.1 · Legal basis](#i1--legal-basis)
- [I.2 · The three components](#i2--the-three-components)
- [I.3 · Sensitivities-based Method (SBM)](#i3--sensitivities-based-method-sbm)
  - [I.3.1 Delta, vega, curvature](#i31--the-three-risk-measures)
  - [I.3.2 The nested aggregation](#i32--the-nested-aggregation)
  - [I.3.3 The three correlation scenarios](#i33--the-three-correlation-scenarios)
  - [I.3.4 GIRR](#i34--girr--general-interest-rate-risk)
  - [I.3.5 CSR non-securitisations](#i35--csr-non-securitisations)
  - [I.3.6 CSR securitisations (non-CTP)](#i36--csr-securitisations-non-ctp)
  - [I.3.7 CSR securitisations (CTP)](#i37--csr-securitisations-ctp)
  - [I.3.8 Equity](#i38--equity)
  - [I.3.9 Commodity](#i39--commodity)
  - [I.3.10 FX](#i310--fx)
- [I.4 · Default Risk Charge (DRC)](#i4--default-risk-charge-drc)
- [I.5 · Residual Risk Add-On (RRAO)](#i5--residual-risk-add-on-rrao)
- [I.6 · Reporting currency & the CRR3 dual run](#i6--reporting-currency--the-crr3-dual-run)
- [I.7 · How the SBM sensitivities are defined](#i7--how-the-sbm-sensitivities-are-defined)

**Part II — Implementation plan**
- [II.1 What exists today](#ii1--what-exists-today)
- [II.2 Package layout](#ii2--package-layout)
- [II.3 Risk-model extensions (`nablatensor-risk`)](#ii3--risk-model-extensions-nablatensor-risk)
- [II.4 SBM engine & per-class modules](#ii4--sbm-engine--per-class-modules)
- [II.5 DRC](#ii5--drc)
- [II.6 RRAO](#ii6--rrao)
- [II.7 SA assembler & output](#ii7--sa-assembler--output)
- [II.8 Regulatory sensitivity extraction](#ii8--regulatory-sensitivity-extraction)
- [II.9 Testing & validation](#ii9--testing--validation)
- [II.10 Sub-phase sequencing & deliverables](#ii10--sub-phase-sequencing--deliverables)
- [II.11 Out of scope / deferred](#ii11--out-of-scope--deferred)
- [References](#references)

---

# Part I — The regulation

## I.1 · Legal basis

| Jurisdiction | Instrument | Notes |
|---|---|---|
| **Basel** | Framework **MAR20** (boundary), **MAR21** (SBM), **MAR22** (DRC), **MAR23** (RRAO); consolidated from BCBS d457 (Jan 2019, "Minimum capital requirements for market risk") | the reference text this doc follows |
| **EU** | **CRR3** (Regulation (EU) 2024/1623), Part Three Title IV Chapter 1a, **Articles 325d–325bp**; the standardised approach is Articles **325e–325ba** (SBM), **325w–325y** (DRC), **325u** (RRAO) | FRTB *reporting* since Jan 2025; **own-funds requirement from 1 Jan 2027**; targeted relief + multiplier 2027–2030 (Phase 4) |
| **UK** | PRA Rulebook — **Market Risk: Advanced Standardised Approach (ASA)** and **Simplified Standardised Approach (SSA)**; PS1/26 final rules | ASA/SSA + trading-book boundary from **1 Jan 2027**; FRTB-IMA from 1 Jan 2028 |

The three jurisdictions share the MAR21/22/23 *structure*; they differ in
parameter values, the list of "liquid" currencies for the GIRR √2 relief, index
look-through rules, and the EU relief package. The code parameterises all of
this (see [II.2](#ii2--package-layout), and Phase 4 `FrtbParameterSet`).

## I.2 · The three components

```
FRTB SA own-funds requirement
    = SBM capital charge      (sensitivities-based: delta + vega + curvature)
    + DRC                     (default risk charge: jump-to-default)
    + RRAO                    (residual risk add-on: gross-notional surcharge)
```

The three are computed independently and **summed** — no diversification between
them. SBM is where essentially all of NablaTensor's adjoint-sensitivity value
sits; DRC and RRAO are notional/position arithmetic with lookup tables.

## I.3 · Sensitivities-based Method (SBM)

The SBM runs over **seven risk classes**:

| # | Risk class | Delta risk factor(s) | Buckets (indicative) |
|---|---|---|---|
| 1 | **GIRR** — general interest-rate risk | risk-free yield curve by currency at 10 tenor vertices; plus inflation and cross-currency basis | one per currency |
| 2 | **CSR non-securitisations** | credit-spread curve per issuer at 5 tenors; bond vs CDS | ~16 (sector × credit quality) + indices |
| 3 | **CSR securitisations (non-CTP)** | tranche credit-spread curve at 5 tenors | ~25 (asset class × seniority × region) |
| 4 | **CSR securitisations (CTP)** | credit spread of the underlying names | ~16 (as CSR non-sec sectors) |
| 5 | **Equity** | spot price per name; **plus equity repo rate** per name/tenor | 13 (size × economy × sector, + "other", + 2 index buckets) |
| 6 | **Commodity** | forward/spot price by maturity and delivery location | 11 (energy, metals, agri, …) |
| 7 | **FX** | exchange rate: reporting currency vs each other currency | one per currency (or per cross for cross-currency options) |

Within each risk class, the bank computes a **delta**, a **vega** and a
**curvature** capital charge, then sums them. The class total is computed under
each of three correlation scenarios; the SBM total is
`Σ_class max(high, medium, low)` — i.e. the max is taken *per risk class*, then
summed (MAR21.6).

### I.3.1 · The three risk measures

**Delta** — linear sensitivity to a delta risk factor `k`:
- Net the sensitivity `s_k` across the whole book (all instruments, all trades).
- Weight it: `WS_k = RW_k · s_k`, with `RW_k` the prescribed risk weight for
  the factor's bucket/tenor.

**Vega** — sensitivity to an *implied volatility* risk factor. The vega risk
factor is the implied vol of an option, indexed by **option maturity** (and, for
GIRR/CSR, additionally by the **residual maturity of the underlying**). Weighted
the same way with a vega risk weight `RW_k` derived from the risk class's
liquidity horizon: `RW_k = min( RW_σ · √(LH_class / 10) , 100% )`.

**Curvature** — extra capital for the second-order (gamma) risk that delta
misses, measured by two *shocked full repricings* per curvature risk factor:

```
CVR_k^+ =  −[ Σ_i ( V_i(x_k + RW_k^curv·x_k) − V_i(x_k) − RW_k^curv·x_k·δ_ik ) ]
CVR_k^- =  −[ Σ_i ( V_i(x_k − RW_k^curv·x_k) − V_i(x_k) + RW_k^curv·x_k·δ_ik ) ]
CVR_k   =  −min( CVR_k^+ , CVR_k^- )         (the worse of the two shocks)
```

where `δ_ik` is instrument `i`'s delta to factor `k` and `RW_k^curv` is the
curvature risk weight (for most classes = the *highest delta risk weight* of the
bucket; for GIRR/FX the shock is a *relative parallel shift*). This is the one
part of SBM that needs repricings, not just sensitivities — two per risk factor.

### I.3.2 · The nested aggregation

For each risk class and each measure (MAR21.4):

```
WS_k  = RW_k · s_k                                     (delta / vega)
        CVR_k                                          (curvature: used directly)

within bucket b:
  K_b = sqrt( max( 0 ,  Σ_k D(WS_k)  +  Σ_{k≠l} R(ρ_kl) · ψ_kl · WS_k · WS_l ) )
  S_b = clamp( Σ_k WS_k , −K_b , +K_b )

across buckets:
  charge = sqrt( max( 0 ,  Σ_b K_b²  +  Σ_{b≠c} R(γ_bc) · ψ_bc · S_b · S_c ) )
```

with

| | delta / vega | curvature |
|---|---|---|
| `D(w)` | `w²` | `max(w, 0)²` |
| `R(c)` | `c` | `c²` |
| `ψ(a,b)` | `1` | `0` iff `a < 0` **and** `b < 0`, else `1` |

The `max(0, …)` under the outer square root is the "alternative specification"
(MAR21.4.7): if the sum is negative, `S_b` is instead
`clamp(Σ WS_k, −K_b, K_b)` already applied — the code keeps the clamp and the
`max(0, …)`, which is what `NestedAggregation` already does.

> **This engine already exists** — `com.nablatensor.risk.NestedAggregation`,
> with the `D/R/ψ` switch and a SIMM concentration hook. Phase 1 does **not**
> rewrite it; it feeds it new parameter tables and risk-factor sets.

### I.3.3 · The three correlation scenarios

Every correlation `ρ` and `γ` is a *medium*-scenario value; the high and low
scenarios transform it (MAR21.6):

| Scenario | Transform | Intuition |
|---|---|---|
| **Medium** | `ρ` (as prescribed) | base |
| **High** | `min(1.25 · ρ, 1)` | correlations rise toward 1 |
| **Low** | `max(2·ρ − 1, 0.75·ρ)` | correlations fall toward 0 / negative |

> Already in `com.nablatensor.risk.CorrelationScenario` — no change.

### I.3.4 · GIRR — general interest-rate risk

**Delta risk factors** (MAR21, EU CRR3 Art 325ap):
- The risk-free yield curve of each currency, at the **10 vertices**
  `0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30` years. The bank may model one curve
  per currency (OIS) or several (per tenor/index); a common simplification is a
  single curve.
- **Inflation risk** — one risk factor per currency, flat across the curve
  (captures the inflation component of index-linked instruments).
- **Cross-currency basis** — one (or two) risk factor(s) per currency, not
  tenor-dependent.

**Bucket** = currency.

**Delta risk weights** — a per-vertex table (indicative, MAR21.42; verify):

| Vertex | 0.25y | 0.5y | 1y | 2y | 3y | 5y | 10y | 15y | 20y | 30y |
|---|---|---|---|---|---|---|---|---|---|---|
| RW | 1.7% | 1.7% | 1.6% | 1.3% | 1.2% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% |

Inflation and cross-currency basis each carry a single flat RW (≈1.6%). For the
**specified liquid currencies** — EUR, USD, GBP, AUD, JPY, SEK, CAD, plus the
bank's domestic currency — every GIRR delta RW **may be divided by √2**.

**Within-bucket correlations** (MAR21.46–48):
- Same curve, different vertices `T_k, T_l`:
  `ρ_kl = max( exp( −θ · |T_k − T_l| / min(T_k, T_l) ) , 40% )`, with `θ = 3%`.
- Different curves (same or different vertex): multiply the above by `0.999`.
- Inflation vs any yield-curve vertex: `40%`.
- Cross-currency basis vs anything else: `0%`.

**Across-bucket correlation** `γ = 50%` for every currency pair.

**Curvature** — the shock is a **relative parallel shift** of *all* vertices of
the currency's curve by the largest GIRR delta RW; up and down; no √2 relief.

**Vega** — implied-vol risk factors on the option-maturity grid
`0.5, 1, 3, 5, 10` years; GIRR liquidity horizon `LH = 60`; vega correlations
use option-maturity distance (and underlying-maturity distance).

### I.3.5 · CSR non-securitisations

**Delta risk factors:** the credit-spread curve of each issuer at tenors
`0.5, 1, 3, 5, 10` years, split into **bond** and **CDS** curves (a basis pair).

**Buckets** (MAR21.53, indicative): ~16 buckets by **sector × credit quality** —
investment-grade buckets 1–8 (sovereigns & central banks; regional/local
government & govt-backed non-financials; financials incl. govt-backed;
basic materials/energy/industrials; consumer goods/transport; tech/telecom;
healthcare/utilities/professional; covered bonds), high-yield & non-rated
buckets 9–15 mirroring the sectors, plus an "other sector" bucket. Qualifying
**indices** get their own buckets.

**Risk weights:** per bucket, from ~0.5% (IG sovereign) to ~12% (HY) — table per
MAR21.55.

**Within-bucket correlation** is a product:
`ρ_kl = ρ_name · ρ_tenor · ρ_basis`, with `ρ_name = 35%` (different issuers, same
bucket; `1` if same issuer), `ρ_tenor = 65%` (different tenors; `1` if same),
`ρ_basis = 99.90%` (bond vs CDS; `1` if same type).

**Across-bucket correlation** `γ_bc`: a matrix keyed by the bucket pair
(broadly ~50% within IG, lower to HY, `0` to the "other" bucket).

**Curvature:** shock = the bucket's risk weight applied to the credit-spread
curve (parallel), up and down. **Vega:** option-maturity grid; `LH` per MAR21.

### I.3.6 · CSR securitisations (non-CTP)

Same shape as CSR non-sec but:
- Risk factor = the **tranche** credit-spread curve at `0.5, 1, 3, 5, 10` years.
- **~25 buckets** by asset class (RMBS, CMBS, consumer ABS, corporate CLO, …) ×
  seniority (senior / non-senior) × region.
- Its own RW / correlation tables; several bucket pairs get `γ = 0`
  (no diversification).

### I.3.7 · CSR securitisations (CTP — correlation trading portfolio)

- **~16 buckets** aligned with the CSR non-sec sectors.
- Risk factor = credit spread of the **underlying constituent names**.
- Recognises **index hedging** within the aggregation; distinct correlation set.
- Interacts with **DRC-CTP** ([I.4.4](#i44--drc-securitisations--ctp)).

### I.3.8 · Equity

**Already implemented** for delta + vega + curvature spot risk
(`com.nablatensor.reg.frtb.FrtbSaSbm`, `EquitySbmParameters`). Phase 1 adds:

- **Equity repo-rate** risk factor — per name, per tenor; its own RW table;
  `ρ(spot, repo of same issuer) = 99.90%`.

For reference, the existing parameters (MAR21, equity spot):
- 13 buckets; delta RW `55 60 45 55 | 30 35 40 50 | 70 50 | 70 | 15 25` %.
- Within-bucket delta ρ: buckets 1–4 = 0.15, 5–8 = 0.25, 9 = 0.075, 10 = 0.125,
  11 = 0 (names stand alone), 12–13 = 0.80.
- Across-bucket γ: both in 1–10 → 0.15; either is 11 → 0; buckets 12 & 13 → 0.75;
  else 0.45.
- Vega RW `min(0.55·√(LH/10), 1)`, `LH = 20` (large cap) or `120` (small
  cap / other).

### I.3.9 · Commodity

**Delta risk factors:** the forward/spot price of each commodity by **maturity**
(`0, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30` years) and **delivery location**.

**Buckets** (MAR21.82, indicative): 11 — (1) energy/solid combustibles,
(2) energy/liquid combustibles, (3) energy/electricity & carbon trading,
(4) freight, (5) metals (non-precious), (6) gaseous combustibles,
(7) precious metals, (8) grains & oilseed, (9) livestock & dairy, (10) softs &
other agri, (11) other commodity.

**Correlations** depend on: same commodity vs same bucket different commodity,
tenor distance, and delivery-location match — a three-factor product like CSR.

**Curvature / vega:** as the pattern above.

### I.3.10 · FX

**Delta risk factor:** the exchange rate between the **reporting currency** and
each other currency. For options on a cross of two non-reporting currencies, the
cross rate is also a risk factor.

**Bucket:** one per currency (no sector buckets).

**Risk weight:** a single value (`15%`) for all currency pairs; for the
**specified liquid currency pairs** the RW is `15% / √2`; a further relief
(`/√2`) applies to pairs formed from two specified liquid currencies against a
common third. (Exact list per MAR21.88 / CRR3.)

**Across-bucket correlation** `γ = 60%`.

**Curvature:** relative shock of the exchange rate by the FX risk weight, up and
down. **Vega:** FX liquidity horizon `LH = 40`.

## I.4 · Default Risk Charge (DRC)

Captures **jump-to-default** — the loss if an issuer defaults suddenly, which the
spread-based SBM does not price. Three sub-charges, **summed**:

### I.4.1 · Gross JTD

Per instrument (MAR22.8, MAR22.14):

```
JTD(long)  =  LGD · notional  +  P&L
JTD(short) = −LGD · notional  +  P&L          (P&L = current market value − notional)
```

`LGD` by seniority: senior debt `25%`, non-senior debt `75%`, equity `100%`,
covered bonds `25%` (indicative — verify). Derivatives are decomposed to their
reference-name notional equivalents. Positions with **residual maturity < 1 year**
are scaled by `maturity / 1yr` (with a floor); equity and some others use a
1-year floor.

### I.4.2 · Net JTD (offsetting)

Within an obligor, offset long and short gross JTD:
`net JTD = Σ JTD(long) − Σ |JTD(short)|` per obligor, respecting the maturity
constraint (a short with shorter maturity than the long it hedges gives only
partial offset).

### I.4.3 · Buckets & the hedge-benefit ratio

Three buckets: **corporates**, **sovereigns**, **local governments /
municipalities**. Within each bucket `b`:

```
HBR_b = Σ net JTD(long)  /  ( Σ net JTD(long) + Σ |net JTD(short)| )

DRC_b = max( Σ_i RW_i · net JTD(long)_i  −  HBR_b · Σ_i RW_i · |net JTD(short)_i| , 0 )
```

`RW_i` by credit-quality bucket (MAR22.24, indicative):

| Rating | AAA | AA | A | BBB | BB | B | CCC | Unrated | Defaulted |
|---|---|---|---|---|---|---|---|---|---|
| RW | 0.5% | 2% | 3% | 6% | 15% | 30% | 50% | 15% | 100% |

**Total DRC (non-sec)** `= Σ_b DRC_b` — no diversification across the three
buckets.

### I.4.4 · DRC securitisations / CTP

- **Securitisations (non-CTP):** `RW` = the securitisation-framework risk weight
  of the tranche (SEC-ERBA / SEC-SA / SEC-IRBA), buckets by asset class,
  offsetting only for the *identical* tranche. **Phase 1 takes the tranche RW as
  a caller-supplied input** (the full securitisation-RW calculation is out of
  scope — same boundary as the output floor, Phase 3).
- **CTP:** buckets by index/underlying; recognises hedging *across* buckets with
  prescribed correlations; its own aggregation formula.

## I.5 · Residual Risk Add-On (RRAO)

A blunt gross-notional surcharge for risks the SBM + DRC framework does not
capture (MAR23):

```
RRAO = 1.0% · Σ notional(instruments with an exotic underlying)
     + 0.1% · Σ notional(instruments bearing other residual risks)
```

- **Exotic underlying** (1.0%): longevity, weather, natural catastrophe,
  realised/implied volatility as an underlying, …
- **Other residual risk** (0.1%): gap risk, correlation risk, behavioural risk —
  digital/binary options, barrier options, cliquets/Asians with path-dependence,
  callable/putable bonds with material behavioural optionality, non-vanilla
  prepayment, dividend risk, …

`notional` is **gross** notional. There is no netting and no aggregation formula
— a plain weighted sum.

## I.6 · Reporting currency & the CRR3 dual run

- All charges are computed in the bank's **reporting currency**; FX delta risk
  factors are defined relative to it; per-currency GIRR/DRC inputs are converted
  at the reporting-date rate before aggregation.
- **CRR3 COREP** (templates C 90.xx) reports FRTB own-funds requirements; during
  the transition the EU requires figures on a **relieved** and an **unrelieved**
  basis. Mechanically this is **two runs of the same assembler with two
  parameter sets** — no recompile (it is the scenario-DSL replay pattern). The
  *relief parameters themselves* — the targeted multiplier and the operational
  relief switches — are **Phase 4**; Phase 1 delivers the dual-run plumbing.

## I.7 · How the SBM sensitivities are defined

MAR21.4.1–4.5 defines each delta/vega **as a specific finite difference**, not as
an analytic derivative:

| Risk class | Delta definition | Bump |
|---|---|---|
| GIRR | `[V(r_k + 1bp) − V(r_k)] / 0.0001` | +1 bp absolute on the vertex |
| CSR (all) | `[V(cs_k + 1bp) − V(cs_k)] / 0.0001` | +1 bp absolute on the spread |
| Equity | `[V(1.01·EQ_k) − V(EQ_k)] / 0.01` | +1% relative on spot |
| Equity repo | `[V(r_k + 1bp) − V(r_k)] / 0.0001` | +1 bp absolute |
| Commodity | `[V(1.01·CTY_k) − V(CTY_k)] / 0.01` | +1% relative |
| FX | `[V(1.01·FX_k) − V(FX_k)] / 0.01` | +1% relative |
| Vega (all) | `[V(σ_k·1.01) − V(σ_k)] / (0.01·σ_k) · σ_k` → sensitivity to `σ` | +1% relative on vol, result × vol |

Curvature uses the RW-sized shocks in [I.3.1](#i31--the-three-risk-measures).

This matters for the implementation: the **letter-compliant** sensitivity is a
prescribed-bump revaluation. An **analytic adjoint** Greek (one reverse sweep)
is economically the same to first order and vastly cheaper, but using it for
regulatory numbers needs a documented equivalence and a validation gate. Phase 1
ships **both** — see [II.8](#ii8--regulatory-sensitivity-extraction).

---

# Part II — Implementation plan

## II.1 · What exists today

| Component | Where | State |
|---|---|---|
| Two-level nested aggregation (`D/R/ψ` switch, SIMM concentration hook) | `com.nablatensor.risk.NestedAggregation` | ✅ complete, risk-class-agnostic |
| `RiskFactor` / `RiskMeasure` / `RiskClass` / `Sensitivities` | `com.nablatensor.risk` | ✅ but equity-shaped (`RiskClass` enum has all 7 values; only `EQUITY` wired) |
| Three correlation scenarios | `com.nablatensor.risk.CorrelationScenario` | ✅ complete |
| Equity SA-SBM (delta + vega + curvature, 3 scenarios, binding-scenario report) | `com.nablatensor.reg.frtb.FrtbSaSbm` + `EquitySbmParameters` | ✅ equity spot only (no repo factor) |
| Scenario/replay seam (`setInput` + replay, no recompile) | `com.nablatensor.scenario.ScenarioRunner` / `Scenario` / `Shock` | ✅ — the vehicle for prescribed-bump sensitivities and curvature repricings |
| Adjoint engine, curve bootstrap + analytic Jacobian, model step blocks | `nablatensor-core`, `nablatensor-quant` | ✅ from MVP/Phase 1 |
| ISDA SIMM equity (delta + vega + concentration) | `com.nablatensor.reg.simm` | ✅ — proves the shared aggregation for a second regime |

**Not present:** GIRR / CSR / commodity / FX SBM; securitisation buckets; DRC;
RRAO; the SA assembler; reporting-currency handling; the regulatory
sensitivity-extraction bridge; COREP output.

## II.2 · Package layout

Chosen defaults (from the package-name discussion): existing equity classes
**stay in place** at `com.nablatensor.reg.frtb` as an equity facade; new code
goes under `.sbm.*`; five per-class packages; `extract` gets its own package;
readable names. `*Parameters` classes live **inside their risk-class package**.

| # | Package | Key types |
|---|---|---|
| 1 | `com.nablatensor.risk` *(existing)* | `RiskFactor` (+`tenor2`), **`RiskClassProfile`** SPI, `RiskClass`/`Sensitivities` touch-ups |
| 2 | `com.nablatensor.reg.frtb.sbm` | `SbmCharge` (generic per-class runner), `SbmResult`, `SbmComponent`, `CurvatureRepricing`, the per-class LOW/MEDIUM/HIGH `max` |
| 3 | `com.nablatensor.reg.frtb.sbm.girr` | `GirrSbm`, `GirrParameters`, `GirrVertex` |
| 4 | `com.nablatensor.reg.frtb.sbm.csr` | `CsrNonSecSbm`, `CsrSecSbm`, `CsrSecCtpSbm`, `CsrNonSecParameters`, `CsrSecParameters`, `CsrSecCtpParameters` |
| 5 | `com.nablatensor.reg.frtb.sbm.equity` | `EquityRepo` extension of `EquitySbmParameters`; new `EquitySbm` generic-path entry (delegated to by the facade) |
| 6 | `com.nablatensor.reg.frtb.sbm.commodity` | `CommoditySbm`, `CommoditySbmParameters` |
| 7 | `com.nablatensor.reg.frtb.sbm.fx` | `FxSbm`, `FxSbmParameters` |
| 8 | `com.nablatensor.reg.frtb.drc` | `Jtd`, `DefaultRiskPosition`, `DrcNonSec`, `DrcSec`, `DrcCtp`, `DrcParameters` |
| 9 | `com.nablatensor.reg.frtb.rrao` | `Rrao`, `ResidualRiskPosition`, `ResidualKind` |
| 10 | `com.nablatensor.reg.frtb.sa` | `FrtbSa` (assembler), `CorepMarketRisk`, `ReportingCurrency`, `FrtbSa.dual(...)` |
| 11 | `com.nablatensor.reg.frtb.extract` | `SbmSensitivities` (Route A bump / Route B adjoint), `PrescribedBump`, `RegulatoryVertexMap` |

No new Maven module — all of this is `nablatensor-reg`, which already depends on
`nablatensor-risk`, `nablatensor-quant`, `nablatensor-scenario`.

## II.3 · Risk-model extensions (`nablatensor-risk`)

Generalise the equity-shaped model **without breaking the equity path** (the
existing `FrtbSaSbmTest`, `IsdaSimmTest`, `NestedAggregationTest` must stay green
unchanged — that is the non-breakage proof).

**`RiskFactor`** — keep the record; add `double tenor2` (default `0.0`) for the
GIRR/CSR vega maturity×maturity grid; add typed factories:

```java
RiskFactor.girrDelta(ccy, vertexYears)          // bucket=ccy, name="OIS"/"3M"/"INFL"/"XCCY", tenor=vertex
RiskFactor.girrVega(ccy, optMat, underlyingResidualMat)
RiskFactor.csrDelta(bucket, issuerCurveId, tenor)   // name encodes issuer + bond/CDS
RiskFactor.csrVega(bucket, issuer, optMat)
RiskFactor.equityRepoDelta(bucket, name, tenor)
RiskFactor.commodityDelta(bucket, commodity, tenor, deliveryLocation)
RiskFactor.commodityVega(bucket, commodity, optMat)
RiskFactor.fxDelta(ccyPair)
RiskFactor.fxVega(ccyPair, optMat)
RiskFactor#asCurvature()          // existing — per-factor
RiskFactor#asCurvatureCurve()     // new — collapses the vertex for curve classes (one curvature factor per curve)
```

**`RiskClassProfile`** — the SPI that keeps `NestedAggregation` untouched:

```java
interface RiskClassProfile {
  RiskClass riskClass();
  double deltaRiskWeight(RiskFactor k);
  double vegaRiskWeight(RiskFactor k);
  double curvatureRiskWeight(RiskFactor k);      // the RW-sized shock magnitude
  double withinBucketRho(RiskFactor k, RiskFactor l);   // MEDIUM scenario
  double acrossBucketGamma(String b, String c);        // MEDIUM scenario
  CurvatureShock curvatureShock(RiskFactor k);          // ABSOLUTE_PARALLEL | RELATIVE_PARALLEL | RELATIVE_SPOT
}
```

`EquitySbmParameters` becomes the first `RiskClassProfile`; `GirrParameters`
etc. follow. Correlation-scenario transforms stay applied *outside* the profile
by the `SbmCharge` runner, exactly as `FrtbSaSbm` does today.

**`NestedAggregation`** — expected **no change**. Confirm the GIRR 40%/`θ` decay
floor expresses cleanly inside a `WithinBucketCorrelation` lambda (it does). Add
a `minCorrelation` clamp *only if* profiling shows a class needs it outside the
lambda — prefer not to.

## II.4 · SBM engine & per-class modules

**`com.nablatensor.reg.frtb.sbm.SbmCharge`** — one generic runner:

```java
SbmResult r = SbmCharge.of(profile)          // a RiskClassProfile
    .delta(bookSensitivities)                 // Sensitivities keyed by RiskFactor
    .vega(bookSensitivities)
    .curvature(curvatureInputs)               // List<CurvatureRepricing>
    .compute();                               // runs all 3 scenarios, returns per-scenario + max
```

Internally: for each `CorrelationScenario`, build a `NestedAggregation` with the
profile's `rw` / `rho` / `gamma` wrapped in `sc.apply(...)` (as `FrtbSaSbm`
already does), aggregate delta + vega + curvature, sum, keep the max.

Per-class packages are then thin: a `GirrParameters implements RiskClassProfile`
plus a `GirrSbm` convenience entry that knows the GIRR risk-factor factories.
Once GIRR + two more classes confirm the shape, collapse the per-class entry
points to `SbmCharge.of(profile, ...)` and keep only the `*Parameters` per
package.

**Curvature** for curve classes is driven through
[II.8](#ii8--regulatory-sensitivity-extraction): the extractor produces the two
shocked PVs per curvature factor and hands `SbmCharge` a `CurvatureRepricing`
record (mirror of today's `FrtbSaSbm.CurvatureInput`).

## II.5 · DRC

`com.nablatensor.reg.frtb.drc` — independent of SBM.

```java
record DefaultRiskPosition(String obligor, DrcBucket bucket, Seniority seniority,
                           CreditQuality quality, double notional, double marketValue,
                           double residualMaturityYears, Double tranucheRwOverride) {}

Jtd.gross(position)                 // LGD·notional + P&L, with the maturity scaling
Jtd.net(positionsForObligor)       // long/short offset per obligor, maturity constraint
DrcNonSec.of(positions).compute()  // buckets → HBR → DRC_b → Σ_b
DrcSec / DrcCtp similarly
```

`DrcParameters`: the LGD-by-seniority table, the RW-by-credit-quality table, the
three-bucket definition. `DrcSec` uses `tranucheRwOverride` (caller-supplied);
`DrcCtp` carries its own cross-bucket correlation matrix.

## II.6 · RRAO

`com.nablatensor.reg.frtb.rrao` — trivial:

```java
enum ResidualKind { EXOTIC_UNDERLYING, GAP_RISK, CORRELATION_RISK, BEHAVIOURAL_RISK, DIVIDEND_RISK, OTHER }
record ResidualRiskPosition(String id, double grossNotional, ResidualKind kind) {}
Rrao.of(positions).compute();   // 1.0%·Σ exotic + 0.1%·Σ other
```

`RraoParameters` maps `ResidualKind` → `{1.0%, 0.1%}` per the MAR23 lists.

## II.7 · SA assembler & output

`com.nablatensor.reg.frtb.sa.FrtbSa`:

```java
FrtbSa.Result r = FrtbSa.of(ReportingCurrency.of("EUR", fxRates))
    .sbm(bookSensitivities, curvatureInputs)      // dispatches to all 7 RiskClassProfiles
    .drc(defaultRiskPositions)
    .rrao(residualPositions)
    .compute();

r.sbm();  r.drc();  r.rrao();  r.total();          // total = Σ_class max(H,M,L)  +  DRC  +  RRAO
r.perRiskClass();  r.bindingScenario();  r.corep(); // CorepMarketRisk: C 90.xx-shaped rows
```

`FrtbSa.dual(relievedParams, unrelievedParams)` → two `Result`s from two
parameter sets, one book, no recompile (CRR3 COREP dual — [I.6](#i6--reporting-currency--the-crr3-dual-run)).

## II.8 · Regulatory sensitivity extraction

`com.nablatensor.reg.frtb.extract.SbmSensitivities` — the bridge from a priced
book to a MAR21 `Sensitivities` vector. Two routes:

- **Route A — prescribed bump (default, letter-compliant).** Drive the existing
  `ScenarioRunner` seam: compile the pricing kernel once, then `setInput` +
  replay for each bumped risk factor per [I.7](#i7--how-the-sbm-sensitivities-are-defined).
  Curvature reuses the same seam for the RW-sized shifts. Deterministic, exact to
  the regulation, `O(#risk factors)` revaluations.
- **Route B — adjoint fast path (opt-in, `SbmSensitivities.adjoint()`).** One
  reverse sweep → `dPV/d(model factor)`; a documented linear map through the
  Phase-1 curve-bootstrap Jacobian re-expresses them at the 10 regulatory GIRR
  vertices / the CSR tenor grid / the vega grid. `O(1)` sweeps. **Validated
  against Route A** in the test suite to a tolerance — this is the headline
  "one sweep vs N revaluations" benchmark for the marketing pages.

Both return a `Sensitivities` keyed by the [II.3](#ii3--risk-model-extensions-nablatensor-risk)
`RiskFactor`s, fed straight into `FrtbSa`.

## II.9 · Testing & validation

Mirror `FrtbSaSbmTest` throughout: an **independent** reference aggregation
written fresh in the test (not calling `NestedAggregation`) must reconcile the
calculator for **every** correlation scenario.

| Layer | What |
|---|---|
| Independent oracle | per risk class (GIRR, CSR×3, equity+repo, commodity, FX): fresh nested aggregation, all 3 scenarios |
| Basel / ISDA worked examples | reconcile to the ISDA FRTB-SA unit-test pack and published QIS worked examples, per class |
| DRC hand-calcs | gross JTD + net JTD + HBR + `DRC_b` + `Σ_b` worked by hand for corporate / sovereign / long-short-hedged cases |
| RRAO | trivial weighted sum; a fixed vector |
| Property tests | scale invariance; long/short antisymmetry where it holds; `max(0,·)` boundary; non-negativity; `total` == `max` over scenarios |
| Route A vs Route B | prescribed-bump vs adjoint sensitivities agree to tolerance on the example books; `nablatensor-validate` `BumpCrossCheck` for the extractor |
| Non-breakage | existing equity SBM / SIMM / `NestedAggregation` tests unchanged and green |
| Bit-repro | `nablatensor-validate` `BitReproTest`-style check on the SA assembler for a fixed book |

Runs on the bank-deployable path: `mvn -o -q test`, JDK 25, single-threaded, no
incubator, no GPU, `junit-jupiter` the only external dependency.

## II.10 · Sub-phase sequencing & deliverables

| Sub-phase | Deliverable | Gate |
|---|---|---|
| **1a** | `RiskFactor` (+`tenor2`, factories), `RiskClassProfile` SPI; existing tests green | non-breakage proof |
| **1b** | `GirrSbm` + `GirrParameters` + independent-oracle test | hardest class; proves curve-tenor + basis + curve-shift curvature |
| **1g** *(parallel from 1b)* | `SbmSensitivities` Route A (bump) + Route B (adjoint) + cross-validation | the benchmark artefact |
| **1c** | `CsrNonSecSbm`, equity repo factor, `CommoditySbm`, `FxSbm` + params + tests | mechanical once 1b lands |
| **1d** | `CsrSecSbm`, `CsrSecCtpSbm` + params + tests | securitisation bucket sets |
| **1e** | `drc/` — JTD engine, offsetting, non-sec/sec/CTP + hand-calc tests | independent of SBM |
| **1f** | `rrao/` + `FrtbSa` assembler + reporting-ccy + `CorepMarketRisk` + `dual(...)` | glue + output |
| **1h** | `docs/reg/` expansion (`frtb-sa-sbm.md` → all classes; new `frtb-sa-drc.md`, `frtb-sa-rrao.md`); `nablatensor-examples` multi-class book → full SA charge, one command | public docs + repro |

**Definition of done for Phase 1:** a mixed multi-risk-class book (rates + credit
+ equity + FX + a barrier option) runs `git clone && mvn -o test` green and the
example prints `SBM + DRC + RRAO = total` with the binding scenario, on a bare
laptop.

## II.11 · Out of scope / deferred

| Item | Where it goes |
|---|---|
| Full securitisation risk-weight calc (SEC-SA / SEC-ERBA / SEC-IRBA) | out of scope; DRC-sec takes tranche RW as input |
| FRTB **IMA** (expected shortfall, PLA test, NMRF, stressed calibration) | separate track; prerequisite for Phase 4 |
| EU targeted multiplier + operational relief switches | **Phase 4** (`FrtbOptions`, `FrtbParameterSet`, legacy Basel 2.5 SMM) |
| Trading-book / banking-book boundary classification | **Phase 5** — decides what enters this calc |
| Output floor arithmetic | **Phase 3** — consumes the SA total from here |
| COREP XBRL binding | out of scope — `CorepMarketRisk` is a data shape only |
| Credit-risk / operational-risk RWA | out of scope (not a market-risk analytics problem) |

---

## References

- **Basel Framework** — MAR20 (boundary), MAR21 (SBM), MAR22 (DRC), MAR23 (RRAO);
  `https://www.bis.org/basel_framework/` — consolidated from BCBS d457,
  *Minimum capital requirements for market risk* (Jan 2019).
- **EU CRR3** — Regulation (EU) 2024/1623, Part Three Title IV Chapter 1a
  (Articles 325d–325bp).
- **EU FRTB market-risk relief** — Commission Delegated Regulation C(2026) 3647
  final (targeted operational relief + targeted multiplier, 2027–2030); EBA
  no-action letter, Aug 2026. *(Phase 4.)*
- **UK** — PRA Rulebook, *Market Risk: Advanced Standardised Approach* /
  *Simplified Standardised Approach*; PRA PS1/26, *Implementation of Basel 3.1:
  Final rules* (20 Jan 2026).
- **ISDA** — FRTB Standardised Approach unit tests / benchmarking materials
  (for reconciliation in [II.9](#ii9--testing--validation)).
- Internal: `docs-internal/NABLATENSOR_REGULATORY_DRIVERS_2026.md`,
  `docs-internal/NABLATENSOR_REG_IMPLEMENTATION_PLAN.md`,
  existing `docs/reg/frtb-sa-sbm.md`.
