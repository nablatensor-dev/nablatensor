#!/bin/bash
# Runs every row of the article, one process at a time. VENV must hold the packages in
# requirements.txt; the RustQuant binary is built with `cargo build --release` in rustquant/.
set -u
VENV=${VENV:-venv}
PY="$VENV/bin/python"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TF_CPP_MIN_LOG_LEVEL=3 NUMBA_NUM_THREADS=8 RAYON_NUM_THREADS=8
export LADDERS=${LADDERS:-3} WARMUP=${WARMUP:-2}
row() { echo "=== $* ($(date +%H:%M:%S))"; "$@" 2>&1 | grep -v -E "^(WARNING|I0000|E0000|W0000|#|Warning: Deprecated)"; }

row $PY financepy_bench.py european 1
row $PY financepy_bench.py european 1 greeks
row $PY financepy_bench.py european-kernel 20000000
row $PY financepy_bench.py european-kernel 20000000 greeks
row ./rustquant/target/release/rustquant-ladder european 20000000
row ./rustquant/target/release/rustquant-ladder european 20000000 greeks
row $PY pfhedge_bench.py european 20000000
row $PY pfhedge_bench.py european 20000000 greeks
row $PY tff_bench.py european 20000000
row $PY tff_bench.py european 20000000 greeks
row $PY tff_bench.py european 20000000 cached
row $PY tff_bench.py european 20000000 greeks cached
row $PY qlrisks_bench.py european 2000000
row $PY qlrisks_bench.py european 2000000 greeks
row $PY ql_bench.py european 20000000
row $PY ql_bench.py european 20000000 greeks

row $PY financepy_bench.py asian 200000
row $PY financepy_bench.py asian 200000 greeks
row ./rustquant/target/release/rustquant-ladder asian 200000
row ./rustquant/target/release/rustquant-ladder asian 200000 greeks
row $PY pfhedge_bench.py asian 200000
row $PY pfhedge_bench.py asian 200000 greeks
row $PY tff_bench.py asian 200000
row $PY tff_bench.py asian 200000 greeks
row $PY tff_bench.py asian 200000 cached
row $PY tff_bench.py asian 200000 greeks cached
row $PY qlrisks_bench.py asian 20000
row $PY qlrisks_bench.py asian 20000 greeks
row $PY ql_bench.py asian 200000
row $PY ql_bench.py asian 200000 greeks
echo "=== done ($(date +%H:%M:%S))"
