# Scenario DSL (Seam 6)

*Keywords: scenario ladder monte carlo java, shock grid setinput, risk ladder no recompile, declarative scenarios*

Declare shocks as data; the runner expands them onto `setInput` + replay of an
**already-compiled kernel** — no re-record, no recompile.

## Building blocks (`nablatensor-scenario`)

| type | role |
|---|---|
| `Shock(input, kind, amount)` | `ABSOLUTE` / `RELATIVE` / `ADDITIVE` move of one named input |
| `Scenario(name, shocks)` | a named bundle; `apply(baseMap)` → shocked map |
| `Ladder.of("S0").absolute().from(80).to(120).step(5)` | a 1-D grid over one input |
| `ScenarioSet` | `list(...)`, `ladder(l)`, `grid(l1, l2, ...)` (cartesian product) |
| `ScenarioRunner` | `run(MonteCarlo, base, set, …)` → `Map<name, Pricing>`; `run(MultiOutput, …)`; `ladder(...)` → `LadderResult{x, price, delta}` |

```java
try (MonteCarlo mc = MonteCarlo.of(Products.europeanCall())
        .market(EquityMarket.atmOneYear()).steps(252).greeks().on("cpu-jit").build()) {

    // one build, a full spot ladder
    var ladder = ScenarioRunner.ladder(mc, EquityMarket.atmOneYear(),
        Ladder.of("spot").absolute().from(90).to(110).step(2), 1_000_000, 42L);

    // named stress scenarios
    var out = ScenarioRunner.run(mc, EquityMarket.atmOneYear(), ScenarioSet.list(
        Scenario.of("base"),
        Scenario.of("crash", Shock.relative("spot", -0.30), Shock.additive("vol", 0.10)),
        Scenario.of("rally", Shock.relative("spot", 0.30))), 1_000_000, 42L);
}
```

## Verification (`ScenarioTest`)

- Each shock kind applies with the right arithmetic; a multi-shock scenario
  composes.
- A ladder and a 2-D grid expand to the expected counts; every grid scenario
  carries one shock per axis.
- A spot ladder re-priced through `setInput` is monotone, and its central
  difference at the money matches the adjoint delta there to `5e-3`.

The FRTB curvature charge is exactly this pattern: two `RELATIVE` shocks (`±RW·x`)
per risk factor, re-priced on the compiled kernel — see
[FRTB SA-SBM](../reg/frtb-sa-sbm.md).
