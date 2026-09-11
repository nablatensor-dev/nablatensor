# NablaTensor side of the ladder

`CrnLadder.java` is the harness behind the `cpu-jit` and `simd` rows: the same
seven-spot ladder as `../ladder.py`, against the Maven Central artefacts
`com.nablatensor:nablatensor-quant:0.1.0` and `nablatensor-simd:0.1.0`.

```bash
mvn -q dependency:build-classpath -Dmdep.outputFile=cp.txt
javac --add-modules jdk.incubator.vector -cp "$(cat cp.txt)" -d out CrnLadder.java
java --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
  -Dengine=cpu-jit -Dthreads=8 -Dnablatensor.crn=on -Dproduct=asian -Dpaths=300000 -Dgreeks=true CrnLadder
```

`-Dengine=simd`, `-Dthreads=1`, `-Dnablatensor.crn=off`, `-Dproduct=european
-Dpaths=20000000` select the other rows; `../mem.sh` runs every configuration
under GNU `time` for the memory column.
