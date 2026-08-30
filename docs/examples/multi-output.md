# One tape, many named risk measures

*Keywords: multiple outputs monte carlo, jacobian adjoint one tape, price plus greeks bundle, multi-output aad*

The engine tape has a single output. `MultiOutput` (in `nablatensor-quant`)
gets `N` named measures out of **one recording and one compiled kernel** with a
selector-weight trick:

- the tape output is `y = Σ_i w_i · f_i(x)`, each `w_i` an extra differentiable
  input;
- `∂y/∂w_i = f_i(x)` — one replay returns every measure's **value**;
- with `w = e_k` (set via `setInput`), `∂y/∂x_j = ∂f_k/∂x_j` — that replay
  returns measure `k`'s full input gradient.

So `1 + N` replays of the same kernel give the values plus the complete `N × M`
Jacobian, and the measures are guaranteed consistent (same tape, same Philox
paths). This is the DoD line *"one tape emits price + ≥3 named risk measures"*.
Native multiple reverse seeds on a single forward sweep is the planned engine
form; this is its exact, backend-agnostic stand-in.

```java
try (MultiOutput mo = MultiOutput.of(rec -> {
        SDouble s0 = rec.input("S0", 100), k = rec.input("K", 100),
                vol = rec.input("sigma", 0.2), r = rec.input("r", 0.03);
        SDouble sT = /* ... GBM path over rec.randn() ... */ s0;
        SDouble disc = r.neg().exp();
        Map<String, SDouble> m = new LinkedHashMap<>();
        m.put("call",     sT.sub(k).max(0.0).mul(disc));
        m.put("digital",  Smooth.gt(rec, sT, k, 1.0).mul(disc));
        m.put("straddle", sT.sub(k).abs().mul(disc));
        return m;
      }).on("cpu-jit").build()) {

    MultiOutput.Result r = mo.run(1_000_000, 42L);
    r.value("digital");                      // digital price
    r.sensitivity("call", "sigma");          // call vega
    r.gradient("straddle");                  // {S0=…, K=…, sigma=…, r=…}
}
```

`MultiOutputTest` checks each value and each Jacobian entry against a standalone
single-output kernel for that measure, at the same seed, to `1e-9`.

**When to use which:**

| | `MultiOutput` | `MultiMetric` |
|---|---|---|
| tapes / compiles | one | one per metric |
| replays per `run` | `1 + N` | `N` |
| measures share the exact same paths | yes | yes (same seed) |
| measures can use different step counts | no | yes |

`Calibrator.leastSquares(...)` uses `MultiOutput` internally: the residual
Jacobian each Levenberg-Marquardt step is one `MultiOutput` evaluation.
