# Five libraries outside the JVM, one ladder

Testing harness and code for the nablatensor.com article *The same ladder
outside the JVM* (https://nablatensor.com/blog/five-libraries-outside-the-jvm-one-ladder): the one-step
European and 252-step Asian tapes on the seven-point spot ladder, in QuantLib,
QuantLib-Risks, tf-quant-finance, financepy, RustQuant and pfhedge, each as it
ships. `results-2026-09-11.log` is the run that produced the article.

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
- `run_all.sh` — every row, one process at a time (~45 min, most of it QuantLib).
- `mem.sh` — the memory column: peak RSS of the RustQuant binary and of the
  NablaTensor `CrnLadder` harness per configuration (`mem-2026-09-11.log` is the
  article's run); the Python harnesses print their own `peak RSS`.

```bash
uv venv --python 3.12 venv && source venv/bin/activate
uv pip install -r requirements.txt
(cd rustquant && cargo build --release)
VENV=venv ./run_all.sh | tee results.log
```

`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is required to import
tf-quant-finance on a current protobuf; `run_all.sh` sets it. The NablaTensor
rows are measured with `nablatensor/CrnLadder.java` against the Maven Central
jars (see `nablatensor/README.md`); `mem.sh` expects `NABLA_CP` to point at that
classpath.
