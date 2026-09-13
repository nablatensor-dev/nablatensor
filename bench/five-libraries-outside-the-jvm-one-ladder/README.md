# Five libraries outside the JVM, one ladder

Testing harness and code for the nablatensor.com article *The same ladder
outside the JVM*
(https://nablatensor.com/blog/five-libraries-outside-the-jvm-one-ladder): the
one-step European and 252-step Asian tapes on the seven-point spot ladder, in
QuantLib, QuantLib-Risks, tf-quant-finance, financepy, RustQuant, pfhedge and
(added after publication) MatLogica AADC, each as it ships.
`results-2026-09-11.log` is the run that produced the article's original
five-library tables.

- `ladder.py` — the ladder itself (warm-up on seed 7, three ladders on seeds
  42–44, seven spots, cold/warm medians, throughput in million paths per
  second per call) shared by the Python harnesses.
- `ql_bench.py` — QuantLib 1.43: `MCEuropeanEngine`, `MCDiscreteArithmeticAPEngine`,
  bump-and-revalue on the same seed. One thread.
- `qlrisks_bench.py` — QuantLib-Risks 1.33.3 (XAD): the same engines with
  adjoint Greeks from one tape; path counts limited by tape memory.
- `tff_bench.py` — tf-quant-finance on TensorFlow 2.21: `GeometricBrownianMotion
  .sample_paths`, `GradientTape`, 8 intra-op threads; `cached` passes a
  `normal_draws` block kept per seed; `xla` sets `jit_compile=True` (slower on CPU).
- `financepy_bench.py` — financepy 1.1.2: Asian `value_mc_fast_vc_numba`; the
  vanilla `value_mc*` methods return NaN in 1.1.2, so `european` times the closed
  form and `european-kernel` calls the serial numba kernel directly (not the
  shipped API — see the article).
- `pfhedge_bench.py` — pfhedge 0.23.0 on torch 2.14 CPU: `BrownianStock` +
  `EuropeanOption`, and a `BaseDerivative` subclass for the arithmetic Asian;
  autograd Greeks; 8 intra-op threads.
- `rustquant/` — RustQuant 0.3.1: `MonteCarloPricer::price_monte_carlo` over
  rayon (`RAYON_NUM_THREADS=8`); no seed argument exists, so bumps are on fresh
  draws.
- `matlogica_bench.py` — MatLogica AADC 1.8.2 (`pip install aadc`, a commercial
  compiler run here under its free 49-day trial — see the article's licensing
  section before reusing these numbers): no built-in Monte Carlo engine, so
  this records the ladder's own GBM/Asian payoff as a scalar computation and
  lets `aadc.evaluate` JIT-compile and replay it across the batch; `cached`
  keeps the `numpy` draw block per seed instead of regenerating it. Needs its
  own venv (see below) — kept apart from the others so the trial-license
  activation doesn't touch an environment shared with anything else.
- `matlogica-vs-simd-investigation.md` — working notes from chasing down the
  MatLogica-vs-`simd` throughput comparison across several runs.
- `run_all.sh` — every row except MatLogica, one process at a time (~45 min,
  most of it QuantLib).
- `mem.sh` — the memory column: peak RSS of the RustQuant binary and of the
  NablaTensor `CrnLadder` harness per configuration (`mem-2026-09-11.log` is the
  article's run); the Python harnesses print their own `peak RSS`.

```bash
uv venv --python 3.12 venv && source venv/bin/activate
uv pip install -r requirements.txt
(cd rustquant && cargo build --release)
VENV=venv ./run_all.sh | tee results.log

# MatLogica, separately:
uv venv --python 3.12 venv-aadc && source venv-aadc/bin/activate
uv pip install aadc numpy
python matlogica_bench.py european 20000000 greeks cached
python matlogica_bench.py asian 200000 greeks cached
```

`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is required to import
tf-quant-finance on a current protobuf; `run_all.sh` sets it. The NablaTensor
rows in the article are the published ones from the draw-cache and Java
articles, measured on the same machine with `CrnLadder.java` on the
[`bench/three-java-quant-libraries-one-ladder`](https://github.com/nablatensor-dev/nablatensor/tree/bench/three-java-quant-libraries-one-ladder)
branch.
