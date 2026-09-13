# The XAD showcase, on NablaTensor

Testing harness and code for the nablatensor.com article *The XAD showcase, on
NablaTensor: Heston, SABR, XVA and LIBOR*
(https://nablatensor.com/blog/xad-showcase-on-nablatensor): the four
quantitative-finance kernels in
[`auto-differentiation/ad-benchmarks`](https://github.com/auto-differentiation/ad-benchmarks)
— the benchmark suite XAD's own README links as evidence for "the fastest
tape library" — reimplemented formula for formula against NablaTensor's
`cpu-jit`/`simd` and MatLogica AADC, with XAD's own C++ tape rebuilt from
source and rerun on the same machine rather than quoted from a different one.

## The four scenarios

- **Heston MC** — stochastic-vol Monte Carlo, European call, 100 Euler steps,
  8 differentiable inputs, 10,000 paths.
- **SABR calibration** — Hagan 2002 implied-vol surface fit, 15 differentiable
  inputs (5 expiries x alpha/rho/nu), no Monte Carlo, wrapped in
  `ad-benchmarks`' own fake 500-iteration optimizer loop.
- **XVA CVA** — 15-swap portfolio, 40 differentiable market inputs, 20
  semi-annual time buckets, 10,000 paths. Keeps one upstream quirk exactly as
  published: the per-bucket CVA discount factor reads the original
  (undiffused) rate curve, and the additive rate-shock reapplies the same
  per-path draw at every bucket — occasionally producing an astronomically
  large but correctly-computed CVA value at these path counts (reproduced
  independently by both NablaTensor and MatLogica; see the article for why).
- **LIBOR market model swaption** — Mike Giles' benchmark
  (`people.maths.ox.ac.uk/~gilesm/codes/libor_AD/testlinadj.cpp`), 161
  differentiable inputs (1 accrual `delta` + 80 forward rates `L0` + 80 vols
  `lambda`), 15 swaptions, 10,000 paths — the widest-input case, and the one
  Giles & Glasserman's *Smoking Adjoints* exists to justify.

Every gradient is checked against a finite-difference bump (or, for SABR,
against the closed-form Hagan formula at the true parameters it was
generated from) before any timing number is trusted — see each script's own
`sanity:` output.

## `nablatensor/` — NablaTensor `cpu-jit` and `simd`

- `HestonBench.java`, `SabrBench.java`, `XvaBench.java`, `LiborBench.java` —
  one class per scenario, `com.nablatensor.engine.SDouble`/`Nabla` against
  `com.nablatensor:nablatensor-quant:0.1.0` and `nablatensor-simd:0.1.0`
  (Maven Central); `-Dengine=cpu-jit|simd -Dthreads=N -Dcached=true`.
- `Bench.java` — shared warmup-then-median timing loop, mirroring
  `ad-benchmarks`' own `src/timing.hpp`.
- `results-2026-09-13.log` — all four benchmarks x `{cpu-jit, simd}` x
  `{1, 8}` threads, cached draws, the run behind the article's tables.
- `xad-upstream-2026-09-13.log` — the actual upstream `ad-benchmarks`
  repository (not vendored here — separate MIT-licensed project, fetched
  fresh), built with only `ENABLE_XAD` on and rerun on this same machine, same
  session, instead of quoting Xcelerit's own published numbers from their
  Xeon Platinum 8488C:
  ```bash
  git clone https://github.com/auto-differentiation/ad-benchmarks
  cd ad-benchmarks
  cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_XAD=ON -DENABLE_CPPAD=OFF -DENABLE_ADEPT=OFF -DENABLE_AUTODIFF=OFF
  cmake --build build -j"$(nproc)"
  ./build/ad_benchmarks   # 10,000 paths, 10 iterations, 3 warmup — its own defaults
  ```
  XAD-Codegen, CppAD, Adept and autodiff stay `ad-benchmarks`' own published
  numbers in the article: XAD-Codegen needs a commercial license this
  reproduction doesn't have, and the other three were left disabled to skip
  dependencies this comparison didn't need.
- `nablatensor/README.md` — the fuller per-file notes, one level down.

```bash
cd nablatensor
mvn -q dependency:build-classpath -Dmdep.outputFile=cp.txt
javac --add-modules jdk.incubator.vector -cp "$(cat cp.txt)" -d out *.java
java --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
    -Dengine=cpu-jit -Dthreads=8 -Dcached=true HestonBench
```

## MatLogica AADC (Python bindings)

- `heston_matlogica.py`, `sabr_matlogica.py`, `xva_matlogica.py`,
  `libor_matlogica.py` — the same four scenarios against `pip install aadc`
  (MatLogica's free trial, same one an earlier article used for the
  seven-spot ladder): each per-path kernel recorded as a scalar computation,
  `aadc.evaluate` JIT-compiles and replays it across a numpy batch,
  `aadc.ThreadPool(8)` for its own parallelism.
- `bench_timing.py` — the shared warmup-then-median loop, same protocol as
  the Java side.
- `results-matlogica-2026-09-13.log` — the run behind the article's MatLogica
  rows.

**These measure the Python bindings, not a bare C++ core.** The free
`pip install aadc` trial ships a compiled Python extension only — no
headers, no static/shared library. AADC's C++ API is real and documented,
but needs an Enterprise or Desk/Machine-Bound license obtained by contacting
MatLogica directly, not a `pip install` — so it isn't measured here, and
every MatLogica number in the article is explicit that it includes
`aadc.evaluate()`'s Python call and array-marshaling overhead on top of the
compiled replay itself.

```bash
uv venv --python 3.12 venv-aadc && source venv-aadc/bin/activate
uv pip install aadc numpy
python heston_matlogica.py
python sabr_matlogica.py
python xva_matlogica.py
python libor_matlogica.py
```

`ad-benchmarks` itself is MIT-licensed (Xcelerit Computing Ltd.); nothing
here copies its C++, only its published formulas, problem sizes and
methodology. The LIBOR model is originally
[Mike Giles'](https://people.maths.ox.ac.uk/~gilesm/codes/libor_AD/testlinadj.cpp),
reproduced by way of `ad-benchmarks`' own adaptation of it.
