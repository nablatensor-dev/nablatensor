# Swap the payoff — the seam in three lines

*Keywords: custom payoff monte carlo java, extensible option pricing library, quant library jvm hackable*

NablaTensor's first design rule: **anything a quant would reasonably want to
tweak lives in a seam.** The payoff is Seam 1 — a `Product` is a functional
interface, you write the valuation in plain Java over `SDouble`, and swapping it
re-records the tape in microseconds. The engine, the `MonteCarlo` driver and the
Greek machinery never change.

Source: [`nablatensor-examples/.../SwapThePayoff.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/SwapThePayoff.java)

## A bespoke payoff, inline

```java
// A capped call: min(max(S_T - K, 0), cap), discounted. Not in the catalogue — three lines.
static Product cappedCall(double cap) {
    return (rec, in, steps) -> {
        SDouble spot = in.of(EquityMarket::spot);
        SDouble strike = in.of(EquityMarket::strike);
        SDouble rate = in.of(EquityMarket::rate);
        SDouble vol = in.of(EquityMarket::vol);
        SDouble maturity = in.of(EquityMarket::maturity);

        GbmPath model = new GbmPath(rec, rate, vol, steps, maturity);
        SDouble s = spot;
        for (int t = 0; t < steps; t++) {
            s = model.step(s, rec.randn());
        }
        SDouble payoff = s.sub(strike).max(0.0).min(rec.constant(cap));
        rec.output(payoff.mul(rate.neg().mul(maturity).exp()));
    };
}
```

Hand it straight to the driver:

```java
try (MonteCarlo mc = MonteCarlo.of(cappedCall(15.0))
        .market(EquityMarket.atmOneYear()).steps(128).greeks().on("cpu-jit").build()) {
    Pricing p = mc.run(2_000_000, 42L);
}
```

## Run it

```bash
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.SwapThePayoff -Dscenarios=2000000 -Dsteps=128
```

## Output

```
128 steps · 2,000,000 scenarios · seed 42 · engine cpu-jit

payoff                      nodes          price        delta         vega
European call                 661       9.419395     0.599447      38.6786
Asian call                    792       5.316334     0.562118      22.4457
Lookback call                 789      17.125289     1.141698      81.9624
Capped call (inline)          663       5.561713     0.272368       2.6038
```

Four different tapes, four `build()`s, no code outside the payoff lambda. The
capped call's Greeks fall out of the same adjoint sweep as everything else —
the cap (a `min` node) is just another op the reverse pass knows how to walk.
