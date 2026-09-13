# The same ladder in C++

Testing harness and code for the nablatensor.com article *The same ladder in
C++: XAD and MatLogica AADC against cpu-jit and simd*
(https://nablatensor.com/blog/the-same-ladder-in-cpp): XAD 2.x driven
directly through its own C++ adjoint-tape API, MatLogica AADC's free Python
trial, and NablaTensor's `cpu-jit`/`simd` (`CrnLadder.java`), all three
re-measured in one sitting, one process at a time. Same NablaTensor harness
as [*the same ladder in three other Java libraries*](https://nablatensor.com/blog/three-java-quant-libraries-one-ladder)
— see that article's own branch,
[`bench/three-java-quant-libraries-one-ladder`](https://github.com/nablatensor-dev/nablatensor/tree/bench/three-java-quant-libraries-one-ladder),
for the fuller per-file notes on `nablatensor/` — included here as a
complete, runnable snapshot rather than this article's own focus.

## `xad/` — XAD's own C++ adjoint-tape API

- `xad_ladder.cpp` — the whole harness: same market, same one-step European
  / 252-fixing arithmetic Asian, same seven-spot ladder,
  warm-up-then-three-ladders-of-seeds-42-44 protocol as the Python/Java
  harnesses. Adjoint Greeks use XAD's own per-path
  `tape.clearAll()`/`registerInput()`/`newRecording()` idiom — the same
  pattern `auto-differentiation/xad`'s own official benchmark
  (`ad-benchmarks/xad/heston_xad.cpp`) uses, one tape recording per path,
  summed and averaged across the batch. One thread by default: neither XAD
  nor its own showcase benchmark threads the tape across paths. `cached`
  generates the draw block once per ladder instead of once per call.
  `threads=N` (N>1) adds a `std::thread` pool *on top* of XAD — one tape per
  thread, paths split into ranges — explicitly not "as it ships," reported
  as a separate, clearly labelled experiment in the article.
- `CMakeLists.txt` — `FetchContent`s XAD from
  `github.com/auto-differentiation/xad` (AGPL-3.0-or-later, no license gate,
  no signup — plain `git clone`), same `-O3 -mavx2 -mfma` flags
  `auto-differentiation/ad-benchmarks` uses for its own XAD rows.

```bash
cd xad
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/xad_ladder european 20000000 greeks          # regenerated draws
./build/xad_ladder european 20000000 greeks cached   # array kept per ladder
./build/xad_ladder asian 300000 greeks cached
./build/xad_ladder asian 300000 greeks cached threads=8   # harness-added pool, not shipped by XAD
```

## `matlogica_bench.py`, `ladder.py` — MatLogica AADC (Python bindings)

MatLogica AADC 1.8.2 (`pip install aadc`, a commercial compiler run under its
free trial — see the article's licensing section before reusing these
numbers): no built-in Monte Carlo engine, so this records the ladder's own
GBM/Asian payoff as a scalar computation and lets `aadc.evaluate`
JIT-compile and replay it across the batch; `cached` keeps the `numpy` draw
block per seed instead of regenerating it. `ladder.py` is the shared
warm-up/three-ladders/cold-warm-median protocol every harness in this
family uses.

```bash
uv venv --python 3.12 venv-aadc && source venv-aadc/bin/activate
uv pip install aadc numpy
python matlogica_bench.py asian 300000 greeks cached
```

## `nablatensor/` — NablaTensor `cpu-jit`/`simd`

- `CrnLadder.java` — the NablaTensor side of the ladder, against
  `com.nablatensor:nablatensor-quant:0.1.0`/`nablatensor-simd:0.1.0` from
  Maven Central, the piece this article re-measures.
- `AnalyticBench.java`, `FinmathBench.java`, `GpuFinmathBench.java`,
  `Greeks.java`, `mem.sh` — the rest of the shared harness (Strata/JQuantLib
  closed forms, finmath-lib CPU/OpenCL, the `cpu-jit` Greek agreement table,
  peak-RSS measurement) from the earlier Java-libraries articles — not this
  article's focus, included for a complete, runnable snapshot.

```bash
cd nablatensor
mvn -q dependency:build-classpath -Dmdep.outputFile=cp.txt
javac --add-modules jdk.incubator.vector -cp "$(cat cp.txt)" -d out *.java
java -cp "out:$(cat cp.txt)" -Dproduct=european -Dpaths=20000000 -Dgreeks=true -Dthreads=8 -Dnablatensor.crn=on -Dengine=simd CrnLadder
```

Watch peak RSS before raising the path count on the Asian tape: XAD's
per-path tape holds nothing between paths, so its memory is flat regardless
of path count, but `simd` and MatLogica both grow with the draw cache.
