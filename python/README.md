# `nablatensor` — Python bridge

A thin JPype wrapper that boots one JDK 25 JVM against a NablaTensor checkout's
compiled `*/target/classes` and re-exports the quant/adjoint Java types
(`EquityMarket`, `ExoticProducts`, `OptionType`, `MonteCarlo`, `Pricing`,
`AadEngines`) so notebooks can drive the exact same code as `demo/*.sh`.

It is a convenience for the examples in [`../notebooks/`](../notebooks/), not a
supported public API. See [`../notebooks/README.md`](../notebooks/README.md) for
setup and usage.

```python
import nablatensor as nt
nt.start()
market = nt.EquityMarket(100.0, 100.0, 0.28, 0.03, 1.0)
mc = (nt.MonteCarlo.of(nt.Products.asianCall())
        .market(market).steps(252).greeks().on(nt.best_engine()).build())
print(mc.run(1_000_000, 42).delta())
```
