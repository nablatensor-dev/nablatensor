# NablaTensor

**Adjoint automatic differentiation for quantitative finance on the JVM. Write the valuation in Java once — get price and every Greek from one reverse sweep, on CPU, SIMD or GPU.**

Record a Monte-Carlo valuation once in plain Java against `SDouble` scalars.
NablaTensor flattens it to a tape and replays that tape — millions of scenarios —
on a generated **bytecode kernel**, on **SIMD**, or on a **GPU**, from the *same*
recording. One forward sweep gives the price; one reverse sweep gives *all*
first-order Greeks, at roughly the cost of the price alone.

- **Apache-2.0**, single repo, no CLA.
- **Clone-and-run.** `mvn -o test` is green on a laptop with no GPU, no native
  library and no incubator flag — the default `cpu-jit` backend is plain Java.
- **Hackable.** Changing a payoff or a model step is a three-line diff, not a fork.

> **Not** a deep-learning framework, a market-data platform, or a certified
> regulatory-capital product. It computes the numbers a regulation asks for;
> sign-off is yours.

---

## The benchmark

Arithmetic-average **Asian call**, 252 daily fixings, fp64, 2,000,000 scenarios,
seed 42. One recording; `value + 5 Greeks` from one adjoint sweep versus the same
five Greeks by central bump-and-revalue (`1 + 2×5` price-only replays).

_Machine: JDK 25.0.1, Linux amd64, 16 vCPU. Reproduce with the command below — your numbers will differ._

| method | replays | wall clock | speedup |
|---|--:|--:|--:|
| **adjoint** — value + 5 Greeks, one reverse sweep | **1** | **1.11 s** | **9.7×** |
| central bump — 1 + 2×5 price-only revaluations | 11 | 10.76 s | 1.0× |

Same 1536-node tape, every backend this box can run — `1e10`-scenario Asian
risk run (`1,000,000 × 10,000`), projected from a `1e6` probe:

| engine | precision | scenarios/s | `1e10` run | runs on |
|---|---|--:|--:|---|
| `cpu` | fp64 | 1.0×10⁶ | 2.7 h | scalar JVM reference (the oracle) |
| `cpu-jit` | fp64 | 1.8×10⁶ | 1.6 h | generated straight-line bytecode kernel, segmented for C2 |
| `simd` | fp64 | 4.0×10⁶ | 42 min | JDK Vector API, 8×fp64/lane (`--add-modules jdk.incubator.vector`) |
| `rocm` | fp32 | 1.3×10⁷ | 13 min | fused forward+adjoint HIP kernel, HIPRTC-compiled |
| `vulkan` | fp32 | 1.6×10⁷ | **10 min** | fused GLSL→SPIR-V compute shader, on-device Philox |
| `cuda` | fp32 | *(no NVIDIA device on this box)* | — | fused forward+adjoint CUDA kernel, NVRTC-compiled |

All backends reproduce the scalar oracle **path-for-path** at equal seed
(`cpu-jit` bit-exact, the rest to reduction/rounding order; `rocm`/`vulkan`
price and delta agree to 5 d.p.). See [`docs/validation.md`](docs/validation.md).
GPU backends gate themselves at runtime — absent a device or its loader they
report unavailable and selection falls back, so `mvn -o test` stays green on a
bare laptop.

**Reproduce:**

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-bench exec:java \
  -Dexec.mainClass=com.nablatensor.bench.Benchmarks \
  -Dscenarios=2000000 -Dsteps=252
```

---

## 20 lines

```java
import com.nablatensor.quant.*;
import com.nablatensor.engine.Nabla;

EquityMarket market = EquityMarket.atmOneYear();          // S0=K=100, sigma=20%, r=3%, T=1y

try (MonteCarlo<EquityMarket> mc = MonteCarlo.of(Products.asianCall())  // Seam 1: swap this line for any payoff
        .market(market)
        .steps(252)                                       // or .timeGrid(TimeGrid.of(t1, t2, ...))
        .greeks()                                         // value + every first-order Greek
        .on("cpu-jit")                                    // or .fastest(), or "simd" / "vulkan"
        .build()) {

    Nabla.TypedValuation<EquityMarket> p = mc.run(1_000_000, /*seed*/ 42L);

    System.out.printf("price %.4f  delta %.4f  vega %.4f  rho %.4f  (± %.4f)%n",
        p.price(), p.greek(EquityMarket::spot), p.greek(EquityMarket::vol),
        p.greek(EquityMarket::rate), p.standardError());

    // Seam 2: move the market on the compiled kernel — no re-record, no recompile.
    var bumped = mc.run(market.withSpot(101.0), 1_000_000, 42L);
}
```

`p.greeks()` returns the gradient as an `EquityMarket` of the same shape
(`p.greeks().spot()` is delta); `p.greek(EquityMarket::spot)` reads one directly.

Run the worked version: `VanillaEuropeanGreeks`, `AsianGreeksBackends`,
`FrtbCurvatureShowcase`, `FrtbFullShowcase`, `SwapThePayoff`,
`HestonSabrCalibration`, `MnistMlp` in `nablatensor-examples`.
Docs under [`docs/examples/`](docs/examples/): vanilla, Asian, swap-the-payoff,
exotic-models, barrier-digital, rates-fx, multi-output, sabr-calibration,
heston-calibration, mnist-mlp, FRTB curvature; plus
[`docs/cookbook/custom-ops.md`](docs/cookbook/custom-ops.md).

Or watch one happen: [`demo/`](demo/) has six narrated jshell sessions —
`greeks-on-gpu.sh` (a barrier note, every Greek from one sweep, 20 M paths on
Vulkan), `adjoint-vs-bump.sh` (one reverse sweep vs eleven revaluations),
`calibrate-a-smile.sh` (a SABR fit with an adjoint gradient), and
`one-tape-every-backend.sh` (one recording replayed on every backend, each
checked against the scalar oracle).

---

## What's in the box (MVP + Phase 1 + Phase 2 slice)

| module | what |
|---|---|
| `nablatensor-core` | `SDouble`, `AadRecorder`, `AadTape`, the `AadEngine` SPI, `AadResult`, Philox plumbing, the shared CUDA-C tape codegen |
| `nablatensor-cpu` | scalar JVM replay — always available, the deterministic **oracle** |
| `nablatensor-jit-cpu` | tape → straight-line JVM **bytecode** via the Class-File API — the LTS-clean default |
| `nablatensor-simd` | JDK Vector API replay — opt-in (`--add-modules jdk.incubator.vector`) |
| `nablatensor-vulkan` | tape → GLSL→SPIR-V fused compute shader, dispatched through the Vulkan loader (FFM) |
| `nablatensor-rocm` | tape → fused HIP kernel, HIPRTC-compiled, for AMD devices (FFM) |
| `nablatensor-cuda` | tape → fused CUDA kernel, NVRTC-compiled (+ `cuda-interp` / `cuda-eager`) (FFM) |
| `nablatensor-tensor` / `-backend-{cuda,rocm,vulkan}` | internal: the low-level device runtimes the GPU replay engines dispatch through |
| `nablatensor-ops` | *(P1)* smoothed `STEP`/`GT`/band indicators, `N(x)` / `erf` / `pow`, a macro-form custom-op registry — all in primitive nodes, exact adjoint, every backend |
| `nablatensor-quant` | `EquityMarket` + `GbmPath` + `Products` (European / Asian / lookback) + `MonteCarlo` + `BlackScholes`; *(P1)* `ExoticProducts` (barrier / digital / cliquet / autocallable), `HestonModel` · `SabrModel` · `LocalVolModel` · `HullWhite1F` · `LmmModel`, `BasketOption`, `Hooks` (antithetic / control-variate), `CurveBootstrap` + analytic Jacobian, `Calibrator` (adjoint-gradient L-BFGS), `MultiMetric` |
| `nablatensor-scenario` | *(P2)* Seam 6 — `Shock` / `Scenario` / `Ladder` / `ScenarioSet` / `ScenarioRunner`: declarative shocks → `setInput` + replay, no recompile |
| `nablatensor-risk` | *(P2)* Seam 7 — `RiskFactor`, `Sensitivities`, `Portfolio` / netting-set composition, `NestedAggregation` (the FRTB/SIMM `√(ΣK² + ΣγSS)` engine), `CorrelationScenario`, `TimeProfile` |
| `nablatensor-validate` | replay on every backend at equal seed, diff vs the oracle, bump cross-check → a text **evidence pack** |
| `nablatensor-examples` | worked demos, each also a test and a docs page |
| `nablatensor-bench` | the reproducible comparison harness above |

Every GPU backend compiles with or without its toolchain present and gates at
runtime through `AadEngine.isAvailable()`; none is a build- or test-time
dependency of the `cpu-jit` path.

### The seams (change without forking)

1. **the payoff/model lambda** — `Product<M>` is a functional interface over any `double`-only market record; write the valuation in `SDouble`.
2. **`setInput`** — market data and model params are re-settable on a compiled kernel (`mc.run(shockedMarket, …)`).
   `.timeGrid(TimeGrid.of(t1, t2, …))` swaps the schedule (non-uniform sampling) without touching the payoff.
3. **custom ops** *(P1)* — `CustomOp.registerUnary(name, macro)`; the code generators pick it up on every backend.
4. **hooks** *(P1)* — `Hooks.antithetic(...)` / `Hooks.controlVariate(...)` wrap a payoff with a transformed draw stream.
5. **model blocks** *(P1)* — subclass a `*Model` step block; override `drift()` / `diffusion()`.
6. **scenario DSL** *(P2)* — `ScenarioRunner.run(kernel, base, ScenarioSet.grid(...), …)`; shocks are data.
7. **aggregation** *(P2)* — record per-trade; `Portfolio.aggregate()` / `byNettingSet()`; compose buckets and correlations outside the tape.
8. **backend choice** — `.on("cpu-jit" | "simd" | "vulkan" | …)` or `.fastest()`; same tape, same numbers.

---

## Roadmap

- **MVP** — engine + `cpu` oracle / `cpu-jit` / `simd` / `vulkan` / `rocm` / `cuda`, vanilla / Asian / lookback, GBM, validation harness, benchmark. ✅
- **Phase 1** — smoothed indicator + `N(x)`/`erf`/`pow` + custom-op registry; barrier / digital / cliquet / autocallable / floating lookback; caps-floors / swaptions (Hull-White + LMM); FX / quanto; correlated basket; Bermudan *shell*; Heston / SABR / local-vol / Hull-White 1F / LMM step blocks with `drift()`/`diffusion()` hooks; antithetic / control-variate / importance-sampling / path-filter hooks; deposit-FRA-swap curve bootstrap + analytic Jacobian; adjoint-gradient calibration (L-BFGS **and** Levenberg-Marquardt), SABR closed-form and Heston Monte-Carlo fits; `MultiOutput` (one tape → N named measures) and `MultiMetric`. ✅
- **Phase 2** — ⬤ *slice done*: aggregation layer (`nablatensor-risk`) and scenario DSL (`nablatensor-scenario`), including the FRTB curvature repricing showcase. ⬡ *remaining*: XVA, VaR / ES, AVA, SA-CCR, IRRBB, prescribed-bump/adjoint sensitivity extraction, multi-GPU.
- **Phase 3** — nested stochastic (Solvency II SCR, VM-22, IFRS 17); LSM early exercise; second-order adjoints.

**Native multiple reverse seeds on one forward sweep** — a tape can now carry
several `rec.output(name, ...)` calls, and one replay runs the forward sweep once
and one reverse sweep per output, returning every value, its standard error and
its full gradient (`MultiOutput` is built on this). Native on `cpu` and
`cpu-jit`; `simd` / `vulkan` / `rocm` / `cuda` decline a multi-output or
`rec.randu()` / `rec.stream(...)` tape and selection falls back.

**Still deferred to an engine change:** a *fused* custom op with a hand-written
`{forward, adjoint}` and per-backend code generation — `CustomOp` is the
composable-macro form, which is what runs on every backend including GPU. GPU
support for multi-output and the extended random surface (`randu`, named
streams). Multi-curve (OIS + tenor basis) curve building and a Bermudan LSM
continuation estimator are Phase 2/3.
GPU backends stay `fp32`. `vs-quantlib.md` is structural — the numbered
comparison needs the QuantLib-Java binding wired into `nablatensor-bench`.

---

## Building

Requires JDK 25+ and Maven 3.9+.

```bash
mvn -o -q test          # green with no GPU, no native lib, no incubator flag
mvn -o -q install
```

## Regulation

- [`docs/reg/frtb-for-dummies.md`](docs/reg/frtb-for-dummies.md) — story-driven introduction to the complete FRTB Standardised Approach
- [`docs/reg/frtb-buckets-and-hedging.md`](docs/reg/frtb-buckets-and-hedging.md) — when within- and cross-bucket positions really reduce capital
- [`docs/reg/frtb-sa-sbm.md`](docs/reg/frtb-sa-sbm.md) — FRTB SA Sensitivities-Based Method, equity
- [`docs/reg/isda-simm.md`](docs/reg/isda-simm.md) — ISDA SIMM, equity
- [`docs/examples/frtb-curvature-showcase.md`](docs/examples/frtb-curvature-showcase.md) — executable shocked-repricing showcase
- [`docs/examples/frtb-curvature-for-beginners.md`](docs/examples/frtb-curvature-for-beginners.md) — illustrated beginner's guide to the calculation
- [`docs/examples/portfolio-aggregation.md`](docs/examples/portfolio-aggregation.md) · [`docs/examples/scenario-dsl.md`](docs/examples/scenario-dsl.md)

## Compare

- [`docs/compare/vs-bump-and-revalue.md`](docs/compare/vs-bump-and-revalue.md) — per-product adjoint-vs-bump table
- [`docs/compare/vs-finmath.md`](docs/compare/vs-finmath.md)
- [`docs/compare/vs-quantlib.md`](docs/compare/vs-quantlib.md)

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
