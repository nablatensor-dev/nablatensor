# The same ladder on a GPU (finmath-lib-opencl-extensions)

Testing harness and code for the nablatensor.com article *The same ladder on
a GPU*
(https://nablatensor.com/blog/same-ladder-on-a-gpu-finmath-opencl): the
one-step European and 252-step Asian tapes on the seven-point spot ladder,
finmath-lib's CPU array factory against its
`net.finmath:finmath-lib-opencl-extensions` OpenCL one, alongside NablaTensor
`cpu-jit`/`simd`. Same harness as
[*the same ladder in three other Java libraries*](https://nablatensor.com/blog/three-java-quant-libraries-one-ladder)
— see that article's own branch,
[`bench/three-java-quant-libraries-one-ladder`](https://github.com/nablatensor-dev/nablatensor/tree/bench/three-java-quant-libraries-one-ladder),
for the fuller per-file notes — with `GpuFinmathBench.java` as the piece this
article adds.

## `nablatensor/`

- `GpuFinmathBench.java` — the same tapes as `FinmathBench.java` through
  `RandomVariableOpenCLFactory` instead of the CPU array factory. Price-only
  works; the AAD Greeks path throws (two different exceptions depending how
  the `BrownianMotion` is wired — see the article). Calls `System.exit(0)`
  explicitly since JOCL leaves a non-daemon thread running otherwise.
- `FinmathBench.java` — finmath-lib on the CPU array factory, the baseline
  this article compares the OpenCL factory against.
- `AnalyticBench.java`, `Greeks.java`, `mem.sh`, `CrnLadder.java` — the rest
  of the shared harness (Strata/JQuantLib closed forms, the `cpu-jit` Greek
  agreement table, peak-RSS measurement, and the NablaTensor side against
  `com.nablatensor:nablatensor-quant:0.1.0`/`nablatensor-simd:0.1.0` from
  Maven Central) — not this article's focus, included for a complete,
  runnable snapshot.

```bash
cd nablatensor
mvn -q dependency:build-classpath -Dmdep.outputFile=cp.txt
javac --add-modules jdk.incubator.vector -cp "$(cat cp.txt)" -d out *.java
java -cp "out:$(cat cp.txt)" -Dproduct=asian -Dpaths=200000 -Dgreeks=false -Dcached=true GpuFinmathBench
```
