# Synthetic CDO tranches and correlation delta

*Keywords: cdo tranche pricing java, gaussian copula java, andersen sidenius basu java, base correlation java, portfolio loss distribution java, correlation delta java*

Feature **F9**. Portfolio credit — the loss distribution of a pool and the
tranches carved out of it — sits in a new `nablatensor-credit` module, separate
from the counterparty-exposure code in `nablatensor-cva`.

Source: [`nablatensor-credit/`](../../nablatensor-credit/src/main/java/com/nablatensor/credit/)
· example [`CdoTrancheShowcase.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/CdoTrancheShowcase.java)

## What's in the module

| Class | What |
|---|---|
| `CreditCurve` | piecewise-flat forward hazard → survival, default probability, parallel-shift |
| `OneFactorGaussianCopula` | conditional default probability given the systemic factor; the Vasicek large-pool loss CDF |
| `PortfolioLossDistribution` | Andersen-Sidenius-Basu recursion (homogeneous LGD) with Gauss-Hermite integration over the factor; `expectedLoss`, `expectedTrancheLoss` |
| `CdoTranche` | `expectedLossFraction`, `parSpread`, `protectionBuyerPv` on a `[attach, detach]` slice |
| `CopulaMarket` / `CopulaMonteCarlo` | the copula as a recorded Monte-Carlo — one adjoint sweep gives the tranche's correlation delta and its default-probability sensitivity |

## Using it

```java
PortfolioLossDistribution dist =
    PortfolioLossDistribution.homogeneous(pd, names, rho, lgd, /*GH nodes*/ 96);
double equityEl = new CdoTranche(0.0, 0.03).expectedLossFraction(dist);

var tape = Nabla.model(new CopulaMarket(rho, pd),
        CopulaMonteCarlo.trancheLoss(0.0, 0.03, names, lgd, T, r, /*width*/ 5e-3))
    .fp64().greeks().on("cpu-jit").build();
var v = tape.run(1_000_000, 42L);
v.greek(CopulaMarket::rho);   // correlation delta (negative for the equity tranche)
```

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.CdoTrancheShowcase
```

## What the tests pin

- The conditional default probability integrates back to the unconditional one.
- The recursion's expected loss equals `PD * LGD` in the large-pool limit, and a
  mezz tranche's EL matches the Vasicek closed form.
- Tranches partition the portfolio loss exactly; the 0-100% tranche EL equals
  the portfolio EL.
- Correlation moves the equity tranche EL down and the senior tranche EL up.
- A par-spread contract has zero PV.
- The recorded copula Monte-Carlo matches the recursion to 6%, and its
  correlation delta matches a central bump to 5% (and is negative for the
  equity tranche).

## Deferred

Base-correlation bootstrap from index tranche quotes, and heterogeneous
notionals / recoveries (a bucketed version of the same recursion).
