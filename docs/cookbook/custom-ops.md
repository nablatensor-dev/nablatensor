# Cookbook: custom operations (Seam 3)

*Keywords: custom operation automatic differentiation java, extend aad engine, register op monte carlo, differentiable macro*

The engine's primitive op set is `+ - * / neg exp log sqrt abs max min` plus
`randn` and constants. Everything a quant payoff needs is a composition of
those, so a "custom op" in NablaTensor is a **macro**: a function that expands
into primitive nodes when it is recorded. Its adjoint is whatever the recorded
sub-graph produces — no special reverse rule, and it runs unchanged on
`cpu-jit`, `simd` and every GPU backend.

## Register and use

```java
import com.nablatensor.ops.CustomOp;
import com.nablatensor.ops.Smooth;

// a soft clamp to [-cap, cap]
CustomOp.registerUnary("softclamp", (rec, x) ->
    Smooth.ramp(rec, x.add(1.0), 0.05).sub(Smooth.ramp(rec, x.sub(1.0), 0.05)).sub(1.0));

SDouble y = CustomOp.unary("softclamp").apply(rec, someScalar);
```

`CustomOp` ships with `relu`, `softplus`, `sigmoid` and `normCdf` pre-registered.

## Building blocks already provided

| need | use |
|---|---|
| smoothed `1{x>0}` / `1{a>b}` / band | `Smooth.step` / `Smooth.gt` / `Smooth.between` |
| smoothed `max(x, 0)` | `Smooth.ramp` |
| normal CDF / PDF / `erf` | `SpecialFn.normCdf` / `normPdf` / `erf` |
| `x^p`, constant or differentiable `p` | `SpecialFn.pow` |

## Verification

`OpsTest` records each op as a one-scenario deterministic function and checks
the value and the adjoint against the closed form, checks that shrinking a
smoothing width recovers the discontinuous limit, and runs a user-registered op
**identically on `cpu`, `cpu-jit`, `simd` and a GPU backend** — the Phase-1
definition-of-done for custom ops.

## What a macro can't do

A genuinely non-composable kernel — one that must call an external special
function not expressible in the primitive set — would need a new `AadOp` value
handled in every code generator (scalar interpreter, batched, bytecode, SIMD,
CUDA-C, GLSL). That fused `{forward, adjoint}` form is a planned engine feature;
in practice the special functions quants reach for (`N(x)`, `erf`, `pow`,
smoothed indicators, softplus, …) are all composable and already here.
