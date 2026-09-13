#!/bin/bash
# Peak resident set (GNU time %M) of every process behind the article's memory columns:
# finmath-lib (FinmathBench, -Xmx16g as in the README), Strata and JQuantLib (AnalyticBench)
# and the NablaTensor CrnLadder harness, one ladder each, one process at a time.
#   CP="out:$(cat cp.txt)" JAVA=java ./mem.sh
set -u
JAVA=${JAVA:-java}
run() { local tag="$1"; shift; /usr/bin/time -f "MEM $tag maxrss_kb=%M" "$@" 2>&1 | grep -E "^MEM|^RESULT" | cut -c1-200; }
for prod in european asian; do
  paths=20000000; [ $prod = asian ] && paths=200000
  for cached in false true; do
    for g in false true; do
      run "finmath $prod cached=$cached greeks=$g" $JAVA -Xmx16g -cp "$CP" \
        -Dproduct=$prod -Dpaths=$paths -Dgreeks=$g -Dcached=$cached -Dladders=1 FinmathBench
    done
  done
done
run "strata closed-form"    $JAVA -cp "$CP" -Dlib=strata AnalyticBench
run "jquantlib closed-form" $JAVA -cp "$CP" -Dlib=jquantlib -Dreps=500000 AnalyticBench
for prod in european asian; do
  paths=20000000; [ $prod = asian ] && paths=300000
  for cfg in "cpu-jit 1 off" "cpu-jit 1 on" "cpu-jit 8 off" "cpu-jit 8 on" "simd 8 off" "simd 8 on"; do
    set -- $cfg
    for g in false true; do
      run "nabla $1 thr=$2 crn=$3 $prod greeks=$g" $JAVA --add-modules jdk.incubator.vector -cp "$CP" \
        -Dengine=$1 -Dthreads=$2 -Dnablatensor.crn=$3 -Dproduct=$prod -Dpaths=$paths -Dgreeks=$g -Dladders=1 CrnLadder
    done
  done
done
echo MEMDONE
