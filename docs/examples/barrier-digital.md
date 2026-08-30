# Barriers, digitals and the smoothed indicator

*Keywords: barrier option greeks monte carlo, digital option delta adjoint, smoothed payoff automatic differentiation, knock-out delta java*

A knock-out delta or a digital delta by bump-and-revalue is notoriously noisy:
the payoff has a jump, so a finite bump straddles it and the estimate has huge
variance. The Phase-1 answer is `nablatensor-ops`' smoothed indicator — a
logistic `STEP` built from primitive nodes — so the whole payoff is
differentiable and **one adjoint sweep gives a genuine (mollified) delta**.

`ExoticProducts` (in `nablatensor-quant`) ships:

| product | contract |
|---|---|
| `barrier(type, UP/DOWN × IN/OUT, level, width)` | single barrier, continuous smoothed monitoring |
| `digitalCash(type, cash, width)` | cash-or-nothing |
| `digitalAsset(type, width)` | asset-or-nothing |
| `cliquet(localFloor, localCap, globalFloor, globalCap, notional)` | ratchet with a global collar |
| `autocallable(level, couponRate, width, notional)` | early-redemption note, smoothed trigger |

```java
Product uo = ExoticProducts.barrier(OptionType.CALL,
        ExoticProducts.Barrier.UP_OUT, /*level*/ 125.0, /*width*/ 1.5);

try (MonteCarlo mc = MonteCarlo.of(uo).market(EquityMarket.atmOneYear())
        .steps(64).greeks().on("cpu-jit").build()) {
    Pricing p = mc.run(200_000, 42L);
    p.delta();     // adjoint knock-out delta, one sweep
}
```

## Verification (`ExoticsTest`)

- **In/out parity, exact:** `UP_OUT + UP_IN == vanilla European` to `1e-9` at the
  same width and seed (the construction is `vanilla × survival` +
  `vanilla × (1 − survival)`).
- **Shrinking width converges:** successive refinements of the smoothing width
  move the knock-out price less and less, settling on the sharp-barrier limit,
  and every value stays in `[0, vanilla]`.
- **Adjoint delta vs bump:** the barrier's adjoint delta matches a central bump
  *of the same smoothed payoff* to `5e-3`.
- **Digital vs closed form:** the cash-or-nothing digital reprices
  `e^{-rT} N(d2)` to `0.03`.
- **Cliquet stays inside its collar**; the autocallable's adjoint delta matches
  its bump.

Run: `mvn -o -q -pl nablatensor-quant test -Dtest=ExoticsTest`

## Multiple named risk measures

`MultiMetric` prices several payoffs off one market and one seed:

```java
try (MultiMetric mm = MultiMetric.market(EquityMarket.atmOneYear()).steps(64)
        .metric("call",      Products.europeanCall())
        .metric("digital",   ExoticProducts.digitalCash(OptionType.CALL, 1.0, 1.0))
        .metric("barrierUO", ExoticProducts.barrier(OptionType.CALL,
                    ExoticProducts.Barrier.UP_OUT, 130.0, 1.0))
        .on("cpu-jit").build()) {
    Map<String, Pricing> r = mm.run(1_000_000, 42L);   // each with its own adjoint gradient
}
```

The engine tape has one output, so this builds one kernel per metric and
replays them at a common seed. A single tape with multiple named outputs — one
forward sweep, N reverse seeds — is the next engine feature; `MultiMetric` is
its drop-in-compatible stand-in.
