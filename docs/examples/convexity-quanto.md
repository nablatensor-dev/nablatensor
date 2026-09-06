# Convexity, timing and quanto adjustments

*Keywords: convexity adjustment java, eurodollar futures convexity java, libor in arrears java, cms convexity adjustment java, quanto adjustment java, timing adjustment java*

Feature **F8**. A rate observed or quoted in one measure but paid in another
needs a correction. This packages the five standard closed forms.

Source: [`nablatensor-quant/.../adjust/`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/adjust/)
· example [`ConvexityQuantoShowcase.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/ConvexityQuantoShowcase.java)

## The adjustments

| Call | Formula | Notes |
|---|---|---|
| `ConvexityAdjustment.eurodollarFutures(a, sigma, t1, t2)` | Hull-White 1F; `-> sigma^2 t1 (t1 + 2 tau) / 2` as `a -> 0` | `forwardRate = futuresRate - adjustment`; also returns `d/d sigma` and `d/d a` |
| `ConvexityAdjustment.inArrears(L0, blackVol, accrual, fixingTime)` | `tau L0^2 (e^{sigma^2 T} - 1) / (1 + tau L0)` | exact for a lognormal forward — a rate paid on its own fixing date |
| `ConvexityAdjustment.cms(y0, blackVol, expiry, paymentsPerYear, numberOfPayments)` | `-0.5 y0^2 sigma^2 T G''(y0)/G'(y0)` | `G` the flat-yield annuity function |
| `TimingAdjustment.liborPaymentShift(L0, blackVol, accrual, fixingTime, payTime)` | in-arrears scaled by `(naturalPay - payTime) / accrual` | pay at fixing recovers the full in-arrears adjustment; pay at natural date gives zero |
| `QuantoAdjustment.quantoForward(m, T)` / `.quantoOption(type, m, T, fixedFx)` | forward `S_0 exp((r_f - rho volS volX) T)`, then `Black76` | zero correlation collapses to the plain foreign option at fixed FX |

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.ConvexityQuantoShowcase
```

## What the tests pin

- Eurodollar futures adjustment reduces to `sigma^2 t1 (t1 + 2 tau) / 2` in the
  Ho-Lee limit, and matches a low-variance Hull-White Monte-Carlo of
  `E^Q[(1/P(t1,t2) - 1)/tau]` to 4%.
- In-arrears closed form equals `tau L0^2 (e^{sigma^2 T} - 1) / (1 + tau L0)`
  exactly, and a measure-change Monte-Carlo agrees.
- `liborPaymentShift` equals the in-arrears value when paid at fixing, is zero
  at the natural date, and flips sign one period beyond it.
- The quanto option matches a Monte-Carlo of `FxProducts.quantoOption` to 2%;
  zero correlation gives the plain foreign option converted at the fixed FX.
- The CMS adjustment is positive and scales with `sigma^2`.
