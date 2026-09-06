# Commodity models and spread options

*Keywords: spark spread option java, kirk approximation java, margrabe exchange option java, schwartz one factor java, commodity mean reversion java, seasonality curve java*

Feature **F10**. Commodity prices mean-revert, and the contracts that matter —
spark and dark spreads, calendar spreads, exchange options — are on the
*difference* of two correlated assets. This adds the Schwartz mean-reverting
model and the spread-option toolkit.

Source: [`nablatensor-quant/.../SchwartzOneFactor.java`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/SchwartzOneFactor.java)
· [`SpreadProducts.java`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/SpreadProducts.java)
· [`analytic/Margrabe.java`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/analytic/Margrabe.java)
· [`analytic/KirkSpreadOption.java`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/analytic/KirkSpreadOption.java)
· example [`SparkSpreadShowcase.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/SparkSpreadShowcase.java)

## What's in the box

| Class | What |
|---|---|
| `Margrabe` | exact exchange option `max(S1 - S2, 0)` — closed form, both deltas |
| `KirkSpreadOption` | spread option `max(S1 - S2 - K, 0)` approximation; `-> Margrabe` as `K -> 0` |
| `SpreadMarket` / `SpreadProducts` | two correlated GBM legs; `spreadOption` / `spreadPut` — one adjoint sweep gives both spot deltas and vegas |
| `SchwartzOneFactor` / `SchwartzMarket` | `d ln S = kappa (level - ln S) dt + sigma dW`, Seam-5 step block; closed-form `futuresPrice` |
| `Seasonality` | a deterministic sum of annual harmonics for a log-forward overlay, with a least-squares `fit` |

## Using it

```java
double kirk = KirkSpreadOption.price(s1, s2, K, vol1, vol2, rho, r, q1, q2, T);

var mc = Nabla.model(spreadMarket, SpreadProducts.spreadOption(K, rho, T, steps))
    .fp64().greeks().on("cpu-jit").build();
// mc.run(...).greeks().s1() and .s2() are the two leg deltas

double futures = SchwartzOneFactor.futuresPrice(schwartzMarket, T);
```

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.SparkSpreadShowcase
```

## What the tests pin

- `KirkSpreadOption` at `K = 0` equals `Margrabe` to `1e-9`.
- The exchange option Monte-Carlo matches `Margrabe` to 2%; the non-zero-strike
  spread MC matches the Kirk approximation to 3%.
- The spread MC adjoint deltas match a central bump to `5e-3`; the first leg is
  long, the second short.
- The Schwartz closed-form futures price matches its simulation to 1%, and the
  far-horizon futures converges to the stationary
  `exp(level + sigma^2 / (4 kappa))`.
- `Seasonality.fit` recovers its coefficients exactly and the function is annual.

## Deferred

The Schwartz-Smith two-factor model (short-term deviation plus a stochastic
equilibrium level) and generalising `BasketOption` beyond three assets are
follow-ups.
