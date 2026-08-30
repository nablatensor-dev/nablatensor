# Validation — the CPU oracle and the evidence pack

*Keywords: model validation monte carlo, SR 11-7 reproducibility, deterministic monte carlo java, independent implementation check*

Every NablaTensor backend replays the **same recorded tape**. The scalar `cpu`
engine is the reference: it walks the tape node by node in plain Java and shares
the Philox random stream with every other backend **path-for-path**. So an
accelerated result can be checked against it at an equal seed with no
statistical allowance — any difference is arithmetic reordering, not noise.

`nablatensor-validate` turns that into a one-call harness:

```java
Report report = ModelValidation.of(Products.asianCall())
    .market(EquityMarket.atmOneYear()).steps(252)
    .scenarios(2_000_000).seed(42L)
    .fp64().tolerance(1e-6)
    .run();

System.out.println(report);           // the evidence pack below
assert report.passed();
```

It does two things:

1. **Backend reproduction.** Replays on every engine this machine can run and
   diffs price and each gradient against the oracle (relative, divided by
   `1 + |oracle|`).
2. **Adjoint cross-check.** Central bump-and-revalue on the oracle, run with
   common random numbers, so the difference against the adjoint gradient is the
   bump's own discretisation error.

## Reproduce

```bash
mvn -o -q install
MAVEN_OPTS="--add-modules jdk.incubator.vector" mvn -o -q -pl nablatensor-validate exec:java \
  -Dexec.mainClass=com.nablatensor.validate.EvidenceMain -Dscenarios=2000000 -Dsteps=252
```

## Evidence pack

```
NablaTensor — model-validation evidence pack
============================================

product        : Asian CALL
market         : S0=100.0000 K=100.0000 sigma=0.2000 r=0.0300 T=1.0000
discretisation : 252 steps
scenarios      : 1,000,000
seed           : 0x000000000000002A
precision      : fp64
tolerance      : 1.00e-06 (relative)
machine        : JDK 25.0.1+8-LTS · Linux amd64 · 16 processors

scalar CPU oracle (reference)
-----------------------------
  price   +5.3058639800e+00   (1,000,000 scenarios in 1.145 s)
  delta   +5.6198939096e-01
  vega    +2.2412432861e+01
  rho     +2.3603384920e+01
  dV/dK   -5.0893075116e-01
  dV/dT   +2.9493448337e+00

backend reproduction vs oracle (equal seed, equal scenarios)
-----------------------------------------------------------
  engine      result      price relΔ       grad relΔ  detail
  rocm        PASS         7.042e-16       3.804e-14  rocm
  simd        PASS         7.042e-15       4.150e-14  simd
  cpu-jit     PASS         0.000e+00       0.000e+00  cpu-jit

adjoint gradient vs central bump-and-revalue on the oracle
---------------------------------------------------------
  bump size      : 5.00e-03 (relative, common random numbers)
  greek              adjoint              bump          absΔ
  delta      +5.61989391e-01   +5.61922391e-01      6.70e-05
  dV/dK      -5.08930751e-01   -5.08905459e-01      2.53e-05
  vega       +2.24124329e+01   +2.24122737e+01      1.59e-04
  rho        +2.36033849e+01   +2.36010273e+01      2.36e-03
  dV/dT      +2.94934483e+00   +2.94935170e+00      6.87e-06

RESULT: PASS — every backend reproduces the oracle within tolerance.
```

`cpu-jit` matches the oracle bit-for-bit; `rocm` (a HIPRTC-compiled GPU kernel)
and `simd` differ only by reduction/rounding order. `vulkan` and
`cuda` are `fp32`-only — pass `.fp32()` to `ModelValidation` to include them; on
this box `vulkan` reproduces the oracle's price and delta to five decimal places
(see the [Asian backend matrix](examples/asian-greeks.md)). `cuda` needs an
NVIDIA device, absent here, so it is skipped rather than failed.

The adjoint gradient agrees with the bump to the bump's own `O(h²)` error —
`rho` is the loosest because the payoff's rate dependence is the most nonlinear
over a `0.5 %` shock.

This is generated automatically from a run and is the shape a model-validation
function (SR 11-7 / TRIM / SS1/23) expects: reproducible inputs, an independent
implementation, seed-for-seed agreement. Widening it into a full submission pack
is Phase 2.
