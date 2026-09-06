# GARCH(1,1) maximum likelihood with an adjoint score

*Keywords: garch java, garch maximum likelihood java, ewma volatility java, riskmetrics lambda java, volatility estimation java, pca yield curve java*

Feature **F4**. Estimating volatilities and correlations is a curriculum chapter
in its own right. The point here is small and specific: the Gaussian negative
log-likelihood of a return series is **recorded once**, and every optimiser
iteration reads the exact score vector — `d(logL)/d(theta)` — from a single
adjoint sweep, the same way a SABR or Heston calibration reads its gradient off
an option surface.

Source: [`nablatensor-quant/.../estimate/`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/estimate/)
· example [`GarchMleShowcase.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/GarchMleShowcase.java)

## What's in the package

| Class | What |
|---|---|
| `Garch11` | `sigma2_t = omega + alpha r_{t-1}^2 + beta sigma2_{t-1}`, fitted by ML; long-run variance, conditional-variance path |
| `Fit` | point estimate + asymptotic standard errors + maximised log-likelihood + optimiser status |
| `Ewma` | the `omega = 0`, `alpha + beta = 1` corner; RiskMetrics `lambda = 0.94`, or fit it |
| `CorrelationEstimator` | sample and EWMA covariance / correlation matrices for a return panel |
| `Pca` | cyclic-Jacobi eigen-decomposition — level / slope / curvature of a curve |

## How the fit is set up

```java
Fit fit = Garch11.fit(returns);   // returns assumed zero-mean

fit.params().alpha();             // ARCH coefficient
fit.persistence();                // alpha + beta
fit.params().longRunVariance();   // omega / (1 - alpha - beta)
fit.standardErrors();             // sqrt of the inverse FD Hessian, order omega/alpha/beta
```

Two implementation choices matter:

- **Conditioning.** The fit runs in coordinates that are all `O(1)` and
  comparably scaled — `omega = omegaFrac * sampleVariance`,
  `persistence = alpha + beta`, `share = alpha / (alpha + beta)` — so the
  adjoint gradient is well conditioned and box bounds alone enforce
  `omega > 0`, `alpha, beta >= 0`, `alpha + beta < 1`. Fitting raw `omega`
  (`~1e-6`, with a gradient `~1e7`) next to `persistence` (`~0.95`) stalls
  L-BFGS on the first step.
- **Backend.** The likelihood unrolls one tape node per observation, so a
  10 000-point series is a tape past what the straight-line bytecode kernel can
  emit — the fit runs on the scalar `cpu` engine, which replays a tape of any
  size.

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.GarchMleShowcase -Dobs=8000
```

## What the tests pin

- A GARCH series simulated with known `(omega, alpha, beta)` is recovered:
  persistence within `0.02`, `alpha`/`beta` within `0.05`, long-run variance
  within 40%, and the fitted log-likelihood is **not below** the
  data-generating parameters' own.
- The MLE point lies within five standard errors of the truth.
- `Ewma.estimateByMaximumLikelihood` agrees with a brute-force grid minimum of
  the likelihood to `0.01` in the decay.
- `Pca.of` reconstructs its input matrix (`V Lambda V^T`), preserves the trace,
  and returns orthonormal loadings in descending-eigenvalue order.
