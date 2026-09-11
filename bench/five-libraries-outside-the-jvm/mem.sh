#!/bin/bash
# Peak resident set (GNU time %M) of the RustQuant binary and of the NablaTensor CrnLadder
# harness (nablatensor/CrnLadder.java, compiled against the Maven Central jars) for every row of
# the article's memory column; the Python harnesses print theirs (VmHWM) themselves.
#   NABLA_CP="nablatensor/out:$(cat nablatensor/cp.txt)" JAVA=java ./mem.sh
set -u
JAVA=${JAVA:-java}
export LADDERS=1 WARMUP=2 RAYON_NUM_THREADS=8
run() { local tag="$1"; shift; /usr/bin/time -f "MEM $tag maxrss_kb=%M" "$@" 2>&1 | grep -E "^MEM|^RESULT" | sed -E 's/, "ladder0".*//; s/"at100".*//'; }
for p in "european 20000000" "asian 200000"; do
  run "rustquant $p price"  ./rustquant/target/release/rustquant-ladder $p
  run "rustquant $p greeks" ./rustquant/target/release/rustquant-ladder $p greeks
done
for prod in european asian; do
  paths=20000000; [ $prod = asian ] && paths=300000
  for cfg in "cpu-jit 1 off" "cpu-jit 1 on" "cpu-jit 8 off" "cpu-jit 8 on" "simd 8 off" "simd 8 on"; do
    set -- $cfg
    for g in false true; do
      run "nabla $1 thr=$2 crn=$3 $prod greeks=$g" $JAVA --add-modules jdk.incubator.vector -cp "$NABLA_CP" \
        -Dengine=$1 -Dthreads=$2 -Dnablatensor.crn=$3 -Dproduct=$prod -Dpaths=$paths -Dgreeks=$g -Dladders=1 CrnLadder
    done
  done
done
