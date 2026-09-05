# FRTB Curvature: Shocked Full Repricing

*Keywords: FRTB curvature Monte Carlo, shocked full repricing, CVR, common random numbers*

Yes: within the FRTB calculation, curvature is usually the most expensive part.
The expensive operation is not the final formula or bucket aggregation. It is
producing two full portfolio repricings for every curvature risk factor.

This showcase performs that workflow on a path-dependent arithmetic Asian
option without depending on a regulatory-calculator module.

New to FRTB? Start with
[FRTB Curvature, Explained Simply](frtb-curvature-for-beginners.md).

Source:
[`FrtbCurvatureShowcase.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/FrtbCurvatureShowcase.java)

## Calculation

The example represents a short Asian call, for which convexity under a large
spot shock produces a visible positive curvature charge. It performs:

1. One adjoint Monte Carlo run to obtain the base delta.
2. A separate price-only kernel compiled once for curvature repricing.
3. Base, $+30\%$, and $-30\%$ spot repricings on that kernel.
4. The FRTB curvature residual:

$$
\begin{aligned}
R_+ &= PV_+ - PV_0 - \Delta s, \\
R_- &= PV_- - PV_0 + \Delta s, \\
CVR &= -\min(R_+, R_-),
\end{aligned}
$$

where $s=0.30S_0$.

5. `NestedAggregation.curvature(...)` for the final scalar aggregation.

All base/up/down valuations use the same random seed. These common random
numbers substantially reduce Monte Carlo noise in the P&L differences. Market
inputs are changed through `ScenarioRunner`, so neither shocked valuation
records nor recompiles the pricing kernel.

## Run

```bash
JAVA_HOME=/path/to/jdk-25-or-newer mvn -q -pl nablatensor-examples -am package

JAVA_HOME=/path/to/jdk-25-or-newer mvn -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.FrtbCurvatureShowcase \
  -Dscenarios=1000000 -Dsteps=252 -Dengine=cpu-jit \
  -DbankFactors=10000
```

Configuration properties:

| Property | Default | Meaning |
|---|---:|---|
| `scenarios` | 1,000,000 | Monte Carlo paths per valuation. |
| `steps` | 252 | Asian-option fixings per path. |
| `seed` | 42 | Common seed for base/up/down. |
| `engine` | `cpu-jit` | NablaTensor execution backend. |
| `bankFactors` | 10,000 | Factor count used only for the sequential timing projection. |

## Reading the timing

The program reports the measured adjoint run, three price-only replays, the
up/down pair, and this deliberately simple projection:

$$
t_{\text{bank,sequential}}
\approx t_{\text{base}} + N_{\text{factors}}(t_{\text{up}}+t_{\text{down}}).
$$

It is a local throughput projection, not a bank SLA. A production run distributes
shocks over workers, reprices heterogeneous portfolios, loads market data, and
incurs orchestration and persistence costs. Those effects can move wall time in
either direction relative to this single-product sequential estimate.

## What is and is not heavy

| Stage | Relative cost |
|---|---|
| Adjoint delta extraction | One pricing run plus one reverse sweep. |
| Curvature input generation | Two full price runs per factor; normally the dominant FRTB stage. |
| CVR formula | A few scalar operations per factor. |
| Correlation aggregation | Pairwise scalar arithmetic over already-netted factors. |
| DRC and RRAO arithmetic | Generally small compared with repricing. |

The example has one market risk factor so its final aggregation is intentionally
small. Increase `scenarios` and `steps` to make the pricing workload realistic;
increase `bankFactors` only to change the reported projection.

## Backend comparison

The exact showcase workflow was measured on 2026-09-02 with one million paths,
252 fixings, seed 42, warm-up, and best-of-three timings. The machine has a
6-core/12-thread Intel Xeon E-2276M.

| Backend | Precision | Adjoint delta | Three price replays | Replay speedup | Complete workflow |
|---|---:|---:|---:|---:|---:|
| `cpu` | fp64 | 8.4087 s | 20.9039 s | 1.00x | 29.3125 s |
| `cpu-jit` | fp64 | 5.3097 s | 13.0496 s | 1.60x | 18.3593 s |
| `simd` | fp64 | 3.5643 s | 6.2855 s | 3.33x | 9.8498 s |

All backends above use fp64 and agree on a CVR of `11.560110`.

Reproduce the comparison with:

```bash
MAVEN_OPTS="--add-modules jdk.incubator.vector" mvn -q -pl nablatensor-bench -am install -DskipTests

MAVEN_OPTS="--add-modules jdk.incubator.vector" mvn -q -pl nablatensor-bench exec:java \
  -Dexec.mainClass=com.nablatensor.bench.CurvatureBackendRun \
  -Dscenarios=1000000 -Dsteps=252 -Drounds=3
```