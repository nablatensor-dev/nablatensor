# Value at Risk and Expected Shortfall

*Keywords: value at risk java, expected shortfall java, delta normal var java, delta gamma cornish fisher java, historical simulation var java, var backtest kupiec christoffersen java*

Feature **F3**. The pieces were already in the box — a scenario DSL that moves a
compiled kernel and re-prices, and an adjoint sweep that returns the sensitivity
vector — so this is the thin estimator and backtest layer that turns them into
VaR / ES numbers.

Source: [`nablatensor-risk/.../ValueAtRisk.java`](../../nablatensor-risk/src/main/java/com/nablatensor/risk/ValueAtRisk.java)
· [`VarBacktest.java`](../../nablatensor-risk/src/main/java/com/nablatensor/risk/VarBacktest.java)
· example [`VarEsShowcase.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/VarEsShowcase.java)

## The three routes

| Method | Call | Good for |
|---|---|---|
| Historical / full revaluation | `ValueAtRisk.historical(PnlVector, alpha)` | an empirical quantile of a P&L sample — a bootstrapped-scenario revaluation, or a window of realised daily P&L |
| Expected Shortfall | `ValueAtRisk.expectedShortfall(PnlVector, alpha)` | the mean loss in the `(1 - alpha)` tail |
| Delta-normal | `ValueAtRisk.deltaNormal(sensitivities, covariance, alpha, horizonDays)` | `z_alpha * sqrt(s' Sigma s)` — exact for a linear book and a Gaussian move |
| Delta-gamma Cornish-Fisher | `ValueAtRisk.deltaGammaCornishFisher(delta, gamma, covariance, alpha, horizonDays)` | quadratic P&L `delta' x + 0.5 x' Gamma x` — skewed, fat-tailed |

Every figure is a **positive loss** at confidence `alpha` (e.g. `0.99`).
Multi-day numbers scale the one-day standard deviation by `sqrt(horizonDays)`.

### How delta-gamma works

`quadraticCumulants(delta, gamma, sigma)` diagonalises
`Sigma^{1/2} Gamma Sigma^{1/2}` (reusing the Jacobi eigensolver from F4's
`Pca`), which turns the quadratic form into a sum of independent
`a_i y_i + 0.5 b_i y_i^2` terms. The first four cumulants then have closed forms,
and the loss quantile comes from a fourth-order Cornish-Fisher expansion. With
`Gamma = 0` the skew and kurtosis vanish and the result is identical to
`deltaNormal` — a test pins that.

## Backtesting

```java
VarBacktest bt = VarBacktest.of(realisedPnl, varForecast, 0.99);
bt.exceptions();                    // losses worse than the forecast
bt.kupiecPValue();                  // unconditional coverage (proportion of failures)
bt.christoffersenPValue();          // independence of exceptions
bt.conditionalCoveragePValue();     // combined
bt.rejectedAt(0.05);               // true => the model fails coverage
```

The chi-square reference distributions for one and two degrees of freedom have
closed-form survival functions (`2 (1 - Phi(sqrt s))` and `exp(-s/2)`), so there
is no incomplete-gamma dependency.

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.VarEsShowcase -Dwindow=4000
```

## What the tests pin

- Historical VaR and ES on a Gaussian P&L sample match `z sigma` and
  `sigma phi(z) / (1 - alpha)` within sampling error.
- Delta-normal is exact (to `1e-9`) on a two-factor linear book, and scales by
  `sqrt(10)` for a ten-day figure.
- For a linear portfolio, historical VaR converges to the delta-normal number.
- Delta-gamma cumulants match a two-million-path Monte-Carlo of the quadratic
  P&L; the Cornish-Fisher VaR is within ~10% of the empirical quantile.
- The backtest accepts a calibrated forecast and rejects an under-forecast by
  conditional coverage; the exception count is exact.
