# NablaTensor vs finmath-lib

*Keywords: finmath alternative, finmath adjoint aad, java quant library comparison, finmath greeks performance*

[finmath-lib](https://finmath.net) is the reference open-source quant library on
the JVM — broad, mature, well-tested. NablaTensor is not trying to match its
breadth. It targets one thing finmath does not do natively: **record a scalar
valuation once and replay it, adjoint-accelerated, on a bytecode / SIMD / GPU
kernel.**

## Where each fits

| | finmath-lib | NablaTensor (MVP) |
|---|---|---|
| Product / model catalogue | very wide (rates, credit, equity, hybrid) | vanilla / Asian / lookback, GBM — *widening in Phase 1* |
| Greeks | finite difference; AAD via `RandomVariableDifferentiable` | one adjoint sweep, all first-order Greeks, from the recorded tape |
| Execution | `RandomVariable` arrays on the JVM heap | tape → fused kernel: `cpu-jit` (bytecode), `simd`, GPU |
| Determinism / audit | per-model | one scalar CPU oracle, every backend reproduces it path-for-path |
| Curves, calibration, day-count | extensive | *Phase 1* |
| Dependencies | pure Java | pure Java (`cpu-jit`); FFM only at the GPU boundary |

## The honest summary

- **Need the catalogue today** — instruments, curves, conventions, a decade of
  test vectors — use finmath.
- **Need a specific valuation to produce a full Greek vector fast, repeatedly**
  (risk ladders, calibration loops, XVA-style revaluation) — that is what the
  record/replay engine is for, and it is the piece that has no direct JVM
  equivalent.
- **Both** — import a finmath curve or product, price it on the NablaTensor
  engine. That interop path is its own worked example (Phase 1).

## Reproducible comparison

A like-for-like Asian-Greeks bench against `finmath` `RandomVariableDifferentiable`
is Phase 1 (it needs the finmath dependency wired into `nablatensor-bench`).
Until then, the honest artifacts are:

- [vs bump-and-revalue](vs-bump-and-revalue.md) — the cost model adjoint replaces.
- [validation](../validation.md) — seed-for-seed reproduction across backends.

Contributions to this page with real numbers are welcome — publish the harness,
the seeds and the machine.
