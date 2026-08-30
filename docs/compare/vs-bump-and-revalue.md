# NablaTensor vs bump-and-revalue

*Keywords: adjoint aad vs bump and revalue, finite difference greeks cost, monte carlo greeks performance java*

Bump-and-revalue gets a Greek by repricing under a shifted input:
`(V(x+h) - V(x-h)) / 2h`. For `N` risk factors that is `1 + 2N` price-only
Monte-Carlo runs (one-sided: `1 + N`). Adjoint AD gets **all `N`** from one
reverse sweep that costs a small constant on top of the price.

## Method

- Arithmetic Asian call, 252 fixings, fp64, GBM, seed 42.
- `N = 5` Greeks: delta, dV/dK, vega, rho, dV/dT.
- Adjoint: one `MonteCarlo` built with `.greeks()`, one `run()` → value + 5 Greeks.
- Bump: one `MonteCarlo` built with `.priceOnly()`, `1 + 2×5 = 11` runs, central.
- Both on `cpu-jit`, both timed best-of-3, common random numbers.
- Harness: [`nablatensor-bench/.../Benchmarks.java`](../../nablatensor-bench/src/main/java/com/nablatensor/bench/Benchmarks.java).

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-bench exec:java \
  -Dexec.mainClass=com.nablatensor.bench.Benchmarks -Dscenarios=2000000 -Dsteps=252
```

## Result

_JDK 25.0.1, Linux amd64, 16 vCPU. 2,000,000 scenarios._

| method | MC replays | wall clock | relative |
|---|--:|--:|--:|
| **adjoint** — value + 5 Greeks, one reverse sweep | **1** | **1.11 s** | **1.0×** |
| central bump — `1 + 2×5` price-only revaluations | 11 | 10.76 s | 9.7× slower |

The ratio is `~ (1 + 2N) / (1 + adjoint overhead)`. It grows linearly with the
number of risk factors: at `N = 20` the bump grid is `41` revaluations against
the same single adjoint sweep. The two agree on every number —

```
adjoint: price=5.301676 delta=0.561932 vega=22.389375 rho=23.603735
```

— see [validation](../validation.md) for the seed-for-seed diff.

## Every equity product

`price + delta + vega + rho + dV/dK + dV/dT` from one adjoint sweep vs the
`1 + 2×5` central-bump grid, per payoff. 1,000,000 scenarios, 128 steps, seed 42,
`cpu-jit`. Harness: [`nablatensor-bench/.../ProductBench.java`](../../nablatensor-bench/src/main/java/com/nablatensor/bench/ProductBench.java).

```bash
mvn -o -q -pl nablatensor-bench exec:java \
  -Dexec.mainClass=com.nablatensor.bench.ProductBench -Dscenarios=1000000 -Dsteps=128
```

| product | adjoint (1 sweep) | bump (11 revals) | speedup |
|---|--:|--:|--:|
| European call | 0.29 s | 2.63 s | 9.1× |
| Asian call | 0.29 s | 2.72 s | 9.3× |
| Lookback call | 0.32 s | 2.87 s | 9.0× |
| Floating lookback | 0.32 s | 3.03 s | 9.6× |
| Barrier up-and-out | 0.50 s | 4.03 s | 8.1× |
| Digital cash-or-nothing | 0.28 s | 2.79 s | 10.0× |
| Cliquet | 0.44 s | 3.44 s | 7.8× |
| Autocallable | 0.30 s | 2.78 s | 9.3× |

The smoothed path-dependent payoffs (barrier, digital, cliquet, autocallable)
get a *usable* adjoint delta this way — a raw-discontinuity bump of those is
dominated by variance, not just slower.

## When bump still wins

- **One or two Greeks, tiny tape.** The adjoint overhead isn't amortised.
- **A non-differentiable payoff with no smoothing.** A raw digital's adjoint is
  zero a.e.; you need the smoothed-indicator op (Phase 1) or a bump.
