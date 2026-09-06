# Jump-diffusion step blocks: Merton and Kou

*Keywords: merton jump diffusion java, kou double exponential java, jump diffusion monte carlo java, volatility smile jumps java, levy model java*

Feature **F7**. A diffusion alone cannot fit the short-dated volatility smile —
the market prices in the possibility of a gap. These two step blocks add a
compound-Poisson jump to geometric Brownian motion.

Source: [`nablatensor-quant/.../MertonJumpModel.java`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/MertonJumpModel.java)
· [`KouJumpModel.java`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/KouJumpModel.java)
· example [`JumpDiffusionShowcase.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/JumpDiffusionShowcase.java)

## The models

| Model | Jump size | Market record |
|---|---|---|
| `MertonJumpModel` | lognormal, `ln Y ~ N(muJ, deltaJ^2)` | `MertonJumpMarket(spot, strike, vol, rate, maturity, jumpIntensity, jumpMean, jumpVol)` |
| `KouJumpModel` | asymmetric two-sided exponential (`probUp`, `etaUp`, `etaDown`) | `KouMarket(spot, strike, vol, rate, maturity, jumpIntensity, probUp, etaUp, etaDown)` |

Both are Seam-5 step blocks with a static `european(type, maturity, steps)`
factory, priced through `Nabla.model(market, valuation)` like every other model.

## How the jump is recorded

Each step draws a diffusion normal and, from a uniform, a **smoothed
at-most-one-jump indicator** `1{U < lambda dt}` (`nablatensor-ops` `Smooth.lt`).
`P(>= 2 jumps per step)` is `O((lambda dt)^2)`, so this is exact as the
monitoring grid gets fine. The jump *count* is not differentiated; the jump-size
parameters (`jumpMean`, `jumpVol`, or `probUp`, `etaUp`, `etaDown`) and the drift
compensator are, so one adjoint sweep returns the jump-parameter risk alongside
the spot / vol / rate Greeks.

The compensator is `-ln(1 + lambda kappa dt)` (not `-lambda kappa dt`): the
at-most-one-jump factor has expectation `1 + lambda kappa dt`, so this form
makes the discounted spot an exact per-step martingale. A residual `O(0.1%)`
bias from the smoothed indicator is visible only in a high-path put-call-parity
check.

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.JumpDiffusionShowcase -Dpaths=1500000
```

## What the tests pin

- `MertonJumpModel` Monte-Carlo converges to the exact
  `MertonJumpDiffusion` Poisson-series price (feature F2) within 1.5%.
- Both models collapse to Black-Scholes as `jumpIntensity -> 0`.
- Put-call parity holds to the documented smoothing bias.
- Jumps raise the price of an at-the-money call (more total variance).
- The adjoint spot delta matches a central bump to `5e-3`; the `jumpMean`
  adjoint matches to 8%.

## Deferred

Variance-Gamma and Bates (Heston + jumps) need a recordable Gamma subordinator
and a Heston-composed step respectively; both are follow-ups.
