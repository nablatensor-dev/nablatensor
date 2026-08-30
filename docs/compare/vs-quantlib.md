# NablaTensor vs QuantLib

*Keywords: quantlib java alternative, quantlib adjoint greeks, quantlib monte carlo performance, jvm quant library vs quantlib*

[QuantLib](https://www.quantlib.org) is the reference open-source quant library
— C++, enormous coverage, decades of production use, with a Java binding via
SWIG (`QuantLib-Java`) and the XAD-based adjoint fork (`QuantLib-Risks`).
NablaTensor is not chasing its breadth. It targets the combination QuantLib's
Java surface does not give you: **a valuation written in plain Java, recorded
once, replayed adjoint-accelerated on a bytecode / SIMD / GPU kernel.**

## Where each fits

| | QuantLib (+ Java binding) | NablaTensor (Phase 1) |
|---|---|---|
| Instrument / model catalogue | vast | vanilla + Asian + lookback + barrier/digital/cliquet/autocallable; GBM, Heston, SABR, local-vol, HW1F, LMM |
| Greeks | bump; adjoint via the XAD fork (C++) | one adjoint sweep, all first-order Greeks + model-parameter gradient, pure JVM |
| Monte-Carlo execution | C++ paths | recorded tape → fused kernel: `cpu-jit`, `simd`, `vulkan`, `rocm`, `cuda` |
| Calibration | Levenberg-Marquardt, finite-difference Jacobian | recorded objective, adjoint gradient, box-projected L-BFGS |
| Curve bootstrap | full, many conventions | annual single-curve bootstrap **with an analytic `d(zero)/d(quote)` Jacobian** |
| Deployment on the JVM | JNI + native `.so`/`.dll` per platform | pure Java for `cpu-jit`; FFM only at the GPU boundary, no native jar |
| Determinism / audit | per-engine | one scalar CPU oracle every backend reproduces path-for-path |

## The honest summary

- **Need the catalogue and the conventions today** — use QuantLib.
- **Need a specific valuation's full Greek + model-parameter gradient, fast and
  repeatedly** (smile calibration, barrier books, scenario ladders) on the JVM
  with no native toolchain — that is what the record/replay engine is for, and
  the Java-native + adjoint + GPU combination has no QuantLib-Java equivalent.

## Reproducible numbers

A like-for-like Greeks-and-throughput comparison against `QuantLib-Java` needs
the SWIG binding and its native library wired into `nablatensor-bench`; that is
tracked as a follow-up. What is reproducible today:

- [`vs-bump-and-revalue.md`](vs-bump-and-revalue.md) — the cost model adjoint replaces (`~10x` on the Asian-Greeks run).
- [`../validation.md`](../validation.md) — seed-for-seed reproduction across `cpu`, `cpu-jit`, `simd`, `rocm`.
- [`../examples/sabr-calibration.md`](../examples/sabr-calibration.md) — SSE `~1e-25`, parameters recovered, ~1.5 s.
- `mvn -o -q -pl nablatensor-quant test` — `ModelsTest`, `ExoticsTest`,
  `BasketAndCurveTest`, `CalibrationTest`: adjoint-vs-bump and
  adjoint-vs-closed-form checks for every Phase-1 product.

Contributions with real QuantLib numbers — the harness, the seeds, the machine —
are welcome on this page.
