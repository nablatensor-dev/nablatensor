# Rates and FX: caps, swaptions, FX and quanto options

*Keywords: caplet monte carlo java, european swaption adjoint greeks, garman kohlhagen java, quanto option pricing adjoint*

Phase 1 adds interest-rate and FX products on top of the `HullWhite1F` and
`LmmModel` step blocks and two lognormal FX markets.

## Interest rate

| product | where | payoff |
|---|---|---|
| Caplet on the short rate | `HullWhite1F.caplet` | `tau * notional * max(r_T - K, 0)`, path-discounted |
| Zero-coupon bond | `HullWhite1F.zeroCouponBond` | `E[exp(-∫r dt)]` |
| European payer swaption | `HullWhite1F.europeanSwaption` | `annuity * max(swapRate - K, 0)`, swap rate/annuity from analytic `P(T, ·)` given the simulated `r_T` |
| Cap / floor on a 4-forward strip | `LmmModel.capFloor(type, …)` | sum of caplets, each `tenor * max(sign (L_i - K), 0)` at period end |
| Payer / receiver swaption on the strip | `LmmModel.payerSwaption` / `receiverSwaption` | `annuity * max(sign (swapRate - K), 0)` |

Every model parameter (`a`, `b`, `sigma` for Hull-White; `vol`, `corr` for the
LMM) is a differentiable input, so one adjoint sweep returns the rate-model
parameter risk. `NewProductsTest` verifies the LMM cap's vega against a bump and
checks ATM payer ≈ receiver swaption parity.

```java
var val = HullWhite1F.europeanSwaption(/*expiry*/ 1.0, /*periods*/ 4, /*accrual*/ 1.0,
                                       /*steps*/ 48, /*strike*/ 0.03);
try (var pricer = Nabla.model(HullWhiteMarket.base(), val).fp64().greeks().on("cpu-jit").build()) {
    var v = pricer.value().with(HullWhiteMarket.base()).scenarios(200_000).seed(1L).run();
    v.greeks().sigma();   // swaption vega w.r.t. the short-rate vol
}
```

## FX

| product | where | notes |
|---|---|---|
| European FX option | `FxProducts.fxOption` | Garman-Kohlhagen: drift `rateDom - rateForeign`; run returns delta and both rho's |
| Quanto option | `FxProducts.quantoOption` | payoff on a foreign asset settled at a fixed FX rate; the asset's domestic-measure drift carries `-corr · volAsset · volFx` |

`NewProductsTest` reconciles the FX call to the Garman-Kohlhagen closed form to
`3e-3` and checks the quanto delta against a bump.
