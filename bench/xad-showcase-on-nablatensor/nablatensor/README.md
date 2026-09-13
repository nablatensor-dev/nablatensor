# The XAD showcase benchmarks, against NablaTensor

Source for the numbers in *the same four benchmarks, on NablaTensor*
(see the blog post for the exact filename/date). These reproduce, formula for
formula, the four quantitative-finance kernels in
[`auto-differentiation/ad-benchmarks`](https://github.com/auto-differentiation/ad-benchmarks)
— the benchmark suite XAD's own README links as evidence for "the fastest
tape library" — against NablaTensor's `cpu-jit` and `simd` engines instead of
XAD/XAD-Codegen/CppAD/Adept/autodiff. `ad-benchmarks` is MIT-licensed
(Xcelerit Computing Ltd.); nothing here copies its C++, only its published
formulas and problem sizes, reimplemented against `com.nablatensor.engine.SDouble`.

- `HestonBench.java` — Heston stochastic-vol Monte Carlo, European call, 100
  Euler steps, 8 differentiable inputs (S0, K, T, r, v0, kappa, theta, xi;
  rho fixed at -0.7), 10,000 paths, same formula as upstream's `heston.hpp`.
- `SabrBench.java` — Hagan 2002 SABR surface calibration, 5 expiries x 3
  params = 15 differentiable inputs, wrapped in upstream's fake "500-iteration
  optimizer loop" (a fixed perturbation schedule, not a real converging fit —
  see `sabr.hpp`). No Monte Carlo: `request.scenarios(1)`, not NablaTensor's
  1,000,000-scenario default for an MC `Request` — the earlier version of this
  harness left that default in place and every run pinned one CPU core for
  minutes evaluating a million needless replays of a deterministic objective;
  worth knowing if you write your own non-Monte-Carlo `Nabla` model.
- `XvaBench.java` — 15-swap portfolio CVA, 40 differentiable market inputs
  (20 zero rates, 10 hazard rates, 10 vol points), 20 semi-annual time
  buckets, 10,000 paths. Keeps one upstream quirk exactly as published: the
  per-bucket CVA discount factor reads the *original* (undiffused) rate
  curve, not the diffused path state used to price the swaps themselves — see
  `xva_compute_cva` in `xva.hpp`. Also keeps upstream's additive rate-shock
  model, which reapplies the *same* per-path Gaussian shock at all 20 buckets
  (the shock array is never redrawn between buckets) — over 20,000 draws that
  occasionally random-walks a curve point to something like -13, and
  `exp(-rate*t)` in the swap discount factor then produces astronomically
  large (but finite, correctly-computed) values. Confirmed with a standalone
  Python check outside NablaTensor entirely (`/tmp/xva_check*.py` during
  development, not checked in) that this is a property of the published toy
  model, not a NablaTensor artifact — `ad-benchmarks` itself never prints or
  validates the CVA value, only the timing, so nobody using it for its stated
  purpose would ever notice. Read the CVA/gradient magnitudes here as a
  computational-load benchmark, not a real risk number.
- `LiborBench.java` — Mike Giles' LIBOR market model swaption portfolio
  (`people.maths.ox.ac.uk/~gilesm/codes/libor_AD/testlinadj.cpp` by way of
  upstream's `libor_swaption.hpp`), 161 differentiable inputs (1 accrual
  `delta` + 80 forward rates `L0` + 80 vols `lambda`), 15 swaptions, 10,000
  paths. `L0[0]` is never stochastically evolved by the path generator (the
  evolution loop only touches `L[n+1..]` for each time step `n`) but still
  feeds the final discount-back division, so it has a real, checkable
  gradient — reproduced exactly, not "fixed," since that is Giles' own
  algorithm. This is the one benchmark upstream exists specifically to stress
  (many more inputs than bump-and-revalue could afford — Giles & Glasserman,
  *Smoking Adjoints*), so its 161-wide reverse sweep is checked against a
  finite-difference bump on three representative inputs (`delta`, `L0_0`,
  `lambda_0`) before any timing number is trusted.
- `Bench.java` — shared warmup-then-median timing loop, mirroring upstream's
  own `src/timing.hpp` (warmup discarded, median of the timed samples, an
  inner-loop repeat count for kernels too small to time in one shot).
- `results-<date>.log` — the run behind the article's tables, all four
  benchmarks x `{cpu-jit, simd}` x `{1, 8}` threads, cached draws
  (`-Dnablatensor.crn=on`), plus every correctness check's output.

RNG is NablaTensor's own Philox stream, not XAD's `std::mt19937` — bit-exact
draws were never expected to match; only price/objective sanity (Heston's
price and Greek signs, SABR's Hagan-vs-synthetic-market check, XVA's
CVA-non-negativity, LIBOR's finite-difference bumps) is checked.

## The XAD/FD baseline is measured on this machine, not quoted cross-CPU

The first draft of the article compared these NablaTensor numbers against
`ad-benchmarks`' own published XAD/FD results — measured by Xcelerit on an
Intel Xeon Platinum 8488C, not this machine. That's not a fair comparison:
`xad-upstream-<date>.log` is instead a straight rebuild of the *actual*
upstream `auto-differentiation/ad-benchmarks` repository (not vendored here —
it's a separate MIT-licensed project, fetched fresh), with only `ENABLE_XAD`
on (`ENABLE_CPPAD`/`ENABLE_ADEPT`/`ENABLE_AUTODIFF` off, to skip dependencies
this comparison doesn't need), run on the same box as every NablaTensor
number in this directory:

```bash
git clone https://github.com/auto-differentiation/ad-benchmarks
cd ad-benchmarks
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_XAD=ON -DENABLE_CPPAD=OFF -DENABLE_ADEPT=OFF -DENABLE_AUTODIFF=OFF
cmake --build build -j"$(nproc)"
./build/ad_benchmarks --csv results/results.csv   # 10,000 paths, 10 iters, 3 warmup — its own defaults
```

XAD turned out *faster* here than on Xcelerit's Xeon across all four
benchmarks (9-25%) — a single-core clock-speed story (this machine boosts to
5.1 GHz; a many-core Xeon is built for throughput, not single-thread turbo),
not a claim about which chip is "better." XAD-Codegen, CppAD, Adept and
autodiff remain `ad-benchmarks`' own cross-machine numbers because they
either need a commercial license (XAD-Codegen) or extra dependencies this
comparison skipped (CppAD/Adept/autodiff) — every place they appear in the
article or its charts is labeled "different CPU."

## Run it

```bash
mvn -q dependency:build-classpath -Dmdep.outputFile=cp.txt
javac --add-modules jdk.incubator.vector -cp "$(cat cp.txt)" -d out *.java
java --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
    -Dengine=cpu-jit -Dthreads=8 -Dcached=true HestonBench
java --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
    -Dengine=simd -Dthreads=8 SabrBench
java --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
    -Dengine=cpu-jit -Dthreads=1 -Dpaths=10000 XvaBench
java --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
    -Dengine=simd -Dthreads=8 -Dpaths=10000 LiborBench
```

Every class accepts `-Dengine=cpu-jit|simd`, `-Dthreads=N`, and (except
`SabrBench`, which has no Monte Carlo) `-Dcached=true|false` and
`-Dpaths=N`; all four print a correctness sanity line before the `RESULT`
line, and none of them should be trusted if that line looks wrong.
