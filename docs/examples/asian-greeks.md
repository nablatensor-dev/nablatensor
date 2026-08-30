# Arithmetic Asian option Greeks — one tape, every backend

*Keywords: asian option monte carlo java, gpu monte carlo greeks, java vector api monte carlo, adjoint aad asian option*

An arithmetic-average Asian call has no closed form, so its risk is normally a
bump-and-revalue grid. Here it is one recording, one adjoint sweep, replayed on
every backend the machine has — **same tape, same seed, same numbers to
Monte-Carlo noise**; only throughput changes.

Source: [`nablatensor-examples/.../AsianGreeksBackends.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/AsianGreeksBackends.java)

## The code

```java
for (String engine : availableEngines()) {              // cpu, cpu-jit, simd, ...
    try (MonteCarlo mc = MonteCarlo.of(Products.asianCall())
            .market(EquityMarket.atmOneYear()).steps(252).fp64().greeks().on(engine).build()) {
        mc.run(scenarios, seed);                         // warm
        Pricing p = mc.run(scenarios, seed);
        // p.price(), p.delta(), p.vega(), p.scenariosPerSecond()
    }
}
```

## Run it

```bash
# cpu-jit / cpu — nothing extra needed
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.AsianGreeksBackends -Dscenarios=2000000 -Dsteps=252

# add the simd row (opt-in incubator module, in-process so MAVEN_OPTS reaches it)
MAVEN_OPTS="--add-modules jdk.incubator.vector" mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.AsianGreeksBackends -Dscenarios=2000000 -Dsteps=252
```

## Output

```
Asian call · 252 fixings · 2,000,000 scenarios · seed 42 · fp64

engine              price        delta         vega         scen/s  runs on
rocm             5.301676     0.561932      22.3894       2.16e+06  ROCm/HIP · AMD Radeon Graphics · gfx1103 · HIPRTC
simd             5.301676     0.561932      22.3894       3.81e+06  Vector API · S_512_BIT, 8x fp64 / 16x fp32 per vector, batch 32
cpu              5.301676     0.561932      22.3894       9.66e+05  scalar JVM · 16 processors · fp64
cpu-jit          5.301676     0.561932      22.3894       1.77e+06  generated straight-line bytecode kernel · segmented for C2
```

Every engine agrees on price and Greeks to the digits shown; `cpu-jit`
reproduces the scalar `cpu` oracle **bit-for-bit** (see [validation](../validation.md)).
This box has an AMD APU, so `rocm` runs a real HIP GPU kernel at fp64; on an APU
that is only ~1.2× the SIMD path — a discrete card, or the `fp32` shaders below,
is where the GPU pulls ahead.

The `AadEngines.available(...)` filter above is fp64, so `vulkan` and `cuda`
(both `fp32`-only) don't appear. At `fp32` the picture on this box:

| engine | scenarios/s | 1e10-path Asian risk run (projected) |
|---|--:|--:|
| `vulkan` | 1.6×10⁷ | ~10 min |
| `rocm` | 1.3×10⁷ | ~13 min |
| `simd` | 5.0×10⁶ | ~34 min |
| `cpu-jit` | 1.8×10⁶ | ~1.5 h |

```bash
# the fp32 / GPU matrix + the 1e10 projection
MAVEN_OPTS="--add-modules jdk.incubator.vector" mvn -o -q -pl nablatensor-bench exec:java \
  -Dexec.mainClass=com.nablatensor.bench.AsianRiskRun -Dprobe=1000000 -Dsteps=252
```

## What to change

- **Backend:** drop the loop, call `.on("cpu-jit")` or `.fastest()` directly.
- **Fixings:** `.steps(n)` — the average is taken over the simulated fixings.
- **Payoff:** `Products.asianPut()`, or a lambda — see [swap the payoff](swap-the-payoff.md).
