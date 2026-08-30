# Stochastic models as composable step blocks (Seam 5)

*Keywords: heston monte carlo java, sabr monte carlo, hull-white one factor java, libor market model java, adjoint model parameter risk*

Phase 1 adds five model step blocks in `nablatensor-quant`. Each is a small class
that reads its parameters from a typed market record, exposes a `step(...)`
method, and provides a ready `european(...)` / product builder. Because the
parameters are `SDouble` inputs, **one adjoint sweep returns the full model
parameter gradient** — the sensitivities a calibration or risk-attribution loop
consumes — next to the usual spot/rate Greeks.

| model | class / market | differentiable parameters | demo payoff |
|---|---|---|---|
| Heston | `HestonModel` / `HestonMarket` | `v0, kappa, theta, xi, rho` | European (full-truncation Euler) |
| SABR | `SabrModel` / `SabrMarket` | `alpha, beta, rho, nu` | European on the forward |
| Local vol (CEV) | `LocalVolModel` / `LocalVolMarket` | `sigma0, skew` | European; `skew=0` ≡ GBM |
| Hull-White 1F | `HullWhite1F` / `HullWhiteMarket` | `r0, level(b), meanReversion(a), sigma` | zero-coupon bond, caplet |
| LMM (4 forwards) | `LmmModel` / `LmmMarket` | `L1..L4, vol, corr` | payer swaption on the strip |

```java
// Heston European call: price + dPrice/d{v0,kappa,theta,xi,rho, spot,strike,rate}
var market = new HestonMarket(100, 100, 0.02, 0.04, 1.5, 0.04, 0.5, -0.7);
try (var pricer = com.nablatensor.engine.Nabla
        .model(market, HestonModel.european(OptionType.CALL, 1.0, 48))
        .fp64().greeks().on("cpu-jit").build()) {
    var v = pricer.value().with(market).scenarios(200_000).seed(1L).run();
    HestonMarket greeks = v.greeks();   // greeks.v0(), greeks.xi(), greeks.rho(), ...
}
```

## Verification

`ModelsTest` checks, for every model, that the **full adjoint parameter
gradient agrees with a central bump-and-revalue on the same seed**, plus a
degenerate identity per model:

- local vol with `skew = 0` reprices the GBM European to `1e-6`;
- the Hull-White zero-coupon bond price lands in `(0, 1)` with `dP/dr0 < 0`;
- the LMM swaption has a strictly positive vega that matches its bump.

Heston's variance-parameter adjoints (`v0, kappa, theta, xi`) match a bump only
to a few percent — full-truncation Euler's `max(v, 0)` floor is non-smooth on a
measure-zero set; the `spot / rate / strike / rho` adjoints are exact to
Monte-Carlo noise. This is a property of the scheme, documented on
`HestonModel`.

Run: `mvn -o -q -pl nablatensor-quant test -Dtest=ModelsTest`
