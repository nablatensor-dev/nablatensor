# The same ladder in C++ — the NablaTensor side

Testing harness and code for the NablaTensor rows in the nablatensor.com
article *The same ladder in C++: XAD and MatLogica AADC against cpu-jit and
simd* (https://nablatensor.com/blog/the-same-ladder-in-cpp): `CrnLadder.java`
re-measured in the same sitting as that article's XAD and MatLogica numbers,
against `com.nablatensor:nablatensor-quant:0.1.0`/`nablatensor-simd:0.1.0`
from Maven Central. Same harness as
[*the same ladder in three other Java libraries*](https://nablatensor.com/blog/three-java-quant-libraries-one-ladder)
— see that article's own branch,
[`bench/three-java-quant-libraries-one-ladder`](https://github.com/nablatensor-dev/nablatensor/tree/bench/three-java-quant-libraries-one-ladder),
for the fuller per-file notes — included here as a complete, runnable
snapshot rather than this article's own focus.

This article's own C++ (XAD) and Python (MatLogica) harnesses are
[`bench/lib-comparison-cpp`](https://github.com/nablatensor-dev/nablatensor-web/tree/main/bench/lib-comparison-cpp)
and
[`bench/lib-comparison-nonjava`](https://github.com/nablatensor-dev/nablatensor-web/tree/main/bench/lib-comparison-nonjava)
in the site repo — not moved here, since neither depends on the NablaTensor
engine.

## `nablatensor/`

- `CrnLadder.java` — the NablaTensor side of the ladder (spots 98..102, warm-up
  on seed 7, three ladders on seeds 42-44), the piece this article re-measures.
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
