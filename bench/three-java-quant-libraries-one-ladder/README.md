# The same ladder in three other Java libraries

Testing harness and code for the nablatensor.com article *The same ladder in
three other Java libraries*
(https://nablatensor.com/blog/three-java-quant-libraries-one-ladder): the
one-step European and 252-step Asian tapes on the seven-point spot ladder, in
finmath-lib, Strata and JQuantLib, against NablaTensor's `cpu-jit` and
`simd`. The same harness also produced the one-thread `cpu-jit` rows the
[draw-cache article](https://nablatensor.com/blog/draw-cache-common-random-numbers)
adds, and (with `GpuFinmathBench.java`) the finmath-lib OpenCL rows in
[*the same ladder on a GPU*](https://nablatensor.com/blog/same-ladder-on-a-gpu-finmath-opencl)
— see that article's own branch,
[`bench/same-ladder-on-a-gpu-finmath-opencl`](https://github.com/nablatensor-dev/nablatensor/tree/bench/same-ladder-on-a-gpu-finmath-opencl),
for the GPU-specific run.

## `nablatensor/`

- `FinmathBench.java` — finmath-lib: one-step European and 252-step Asian,
  price only or price + AAD Greeks, spot ladder with a fresh or a shared
  (memoised) `BrownianMotion`. Runs on one thread as shipped (`-Dshards=1`).
- `GpuFinmathBench.java` — the same tapes through
  `net.finmath:finmath-lib-opencl-extensions`'s `RandomVariableOpenCLFactory`
  instead of the CPU array factory. Price-only works; the AAD Greeks path
  throws (two different exceptions depending how the `BrownianMotion` is
  wired — see the GPU article). Calls `System.exit(0)` explicitly since JOCL
  leaves a non-daemon thread running otherwise.
- `AnalyticBench.java` — Strata `BlackScholesFormulaRepository` and JQuantLib
  `AnalyticEuropeanEngine`, closed-form valuations per second.
- `Greeks.java` — the full `cpu-jit` Greek set and standard error at spot 100
  for the agreement table.
- `mem.sh` — peak RSS (GNU `time`) of every process behind the tables'
  memory column.
- `CrnLadder.java` — the NablaTensor side (same ladder), against
  `com.nablatensor:nablatensor-quant:0.1.0` (pulls in `cpu-jit`) and
  `nablatensor-simd:0.1.0` from Maven Central. The article's NablaTensor rows
  were measured against a local build of the same sources; the harness
  compiles and runs unchanged against the Central jars.

```bash
cd nablatensor
mvn -q dependency:build-classpath -Dmdep.outputFile=cp.txt
javac --add-modules jdk.incubator.vector -cp "$(cat cp.txt)" -d out *.java
java -Xmx16g -cp "out:$(cat cp.txt)" -Dproduct=asian -Dpaths=200000 -Dgreeks=true -Dcached=true FinmathBench
java -cp "out:$(cat cp.txt)" -Dlib=strata AnalyticBench
java -cp "out:$(cat cp.txt)" -Dlib=jquantlib -Dreps=500000 AnalyticBench
```
