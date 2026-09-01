# FRTB SA — Default Risk Charge

*Keywords: FRTB DRC java, default risk charge jump to default, JTD gross net offsetting, hedge benefit ratio open source, MAR22*

The Default Risk Charge (MAR22) capitalises **jump-to-default** — the loss if an
issuer defaults suddenly, which the spread-based [SBM](frtb-sa-sbm.md) does not
price. It is position arithmetic on top of the same book, not a sensitivity
problem.

> **Calculators, not sign-off.** The LGD-by-seniority and RW-by-credit-quality
> tables are transcribed from MAR22 into `DrcParameters` — verify against your
> rulebook. Same-obligor longs and shorts net **unconditionally** here; MAR22.21
> only permits the short to offset a long of the same or lower seniority — wire
> that in before real use.

## Mechanics

```
gross JTD      = ( LGD · notional + (marketValue − notional) ) · maturityScale
maturityScale  = 1                              for equity (1-year floor)
               = clamp(maturityYears, 0.25, 1)  otherwise
net JTD(obligor) = Σ signed gross JTD of that obligor

per bucket b ∈ {corporates, sovereigns, local government}:
  HBR_b = Σ netJTD_long / ( Σ netJTD_long + Σ |netJTD_short| )
  DRC_b = max( Σ_i RW_i · netJTD_long_i − HBR_b · Σ_i RW_i · |netJTD_short_i| , 0 )

DRC = Σ_b DRC_b        (no diversification across the three buckets)
```

`LGD`: senior 25%, non-senior 75%, covered bond 25%, equity 100%.
`RW` by credit quality: AAA 0.5% · AA 2% · A 3% · BBB 6% · BB 15% · B 30% ·
CCC 50% · unrated 15% · defaulted 100%.

## Use

```java
List<DefaultRiskPosition> book = List.of(
    new DefaultRiskPosition("CORP_A", DrcBucket.CORPORATES, Seniority.SENIOR,
                            CreditQuality.BBB, 3000, 3000, 5),           // long
    new DefaultRiskPosition("CORP_B", DrcBucket.CORPORATES, Seniority.SENIOR,
                            CreditQuality.BB, -800, -790, 3));           // short

DefaultRiskCharge.Result r = DefaultRiskCharge.of(book).compute();
r.total();                       // Σ_b DRC_b
r.perBucket();                   // DRC_b
r.hedgeBenefitRatio();           // HBR_b
```

`FrtbSa.of(...).drc(book)` folds it into the [full SA charge](frtb-sa-sbm.md#the-full-sa-charge).

## Verification (`DrcRraoTest`)

Gross JTD is checked against the formula (including the sub-year maturity
scaling). `drcNonSecReconcilesToAHandWorkedExample` recomputes HBR, the weighted
long / short sums and `DRC_b` for a corporates + sovereigns book **fresh in the
test** and matches the calculator.

## Not in this slice

The seniority constraint on short offsets (MAR22.21); DRC for **securitisations**
(RW = the tranche's securitisation risk weight) and **CTP** (cross-bucket
hedging with prescribed correlations); index decomposition; the deducted-from-
capital exception.
