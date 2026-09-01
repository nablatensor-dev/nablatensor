# FRTB SA — Residual Risk Add-On

*Keywords: FRTB RRAO java, residual risk add-on, exotic underlying gap correlation behavioural risk, MAR23 gross notional*

The Residual Risk Add-On (MAR23) is a blunt gross-notional surcharge for risks
the [SBM](frtb-sa-sbm.md) + [DRC](frtb-sa-drc.md) framework does not capture. No
netting, no aggregation formula — a plain weighted sum.

```
RRAO = 1.0% · Σ grossNotional(exotic underlying)
     + 0.1% · Σ grossNotional(other residual risk)
```

- **Exotic underlying** (1.0%): longevity, weather, natural catastrophe,
  realised volatility as an underlying, …
- **Other residual risk** (0.1%): gap risk (digital / barrier options),
  correlation risk (baskets, best-/worst-of), behavioural risk (callable bonds,
  non-vanilla prepayment), dividend risk, …

## Use

```java
List<Rrao.ResidualRiskPosition> book = List.of(
    new Rrao.ResidualRiskPosition("longevity-swap", 1_000_000, ResidualRiskKind.EXOTIC_UNDERLYING),
    new Rrao.ResidualRiskPosition("digital-1",      2_000_000, ResidualRiskKind.GAP_RISK));

Rrao.Result r = Rrao.of(book).compute();
r.exotic(); r.other(); r.total();
```

`FrtbSa.of(...).rrao(book)` folds it into the [full SA charge](frtb-sa-sbm.md#the-full-sa-charge).

## Verification (`DrcRraoTest`)

`rraoIsAWeightedGrossNotionalSum` checks the 1.0% / 0.1% split and the total on a
fixed three-instrument book.

## Not in this slice

The precise MAR23 instrument lists (the `ResidualRiskKind` enum carries a
representative set); the "no double-count with an SBM/DRC risk factor" carve-out;
the treatment of instruments already in an internal-model-eligible desk.
