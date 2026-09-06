# Term-structure Hull-White: caplets, Jamshidian swaptions, and calibration

*Keywords: hull white one factor java, jamshidian swaption java, hull white calibration java, swaption pricing java, caplet analytic java, ho lee model java*

Feature **F6**. The existing `HullWhite1F` Monte-Carlo step block assumes a flat
initial forward. This adds the term-structure-consistent analytic model: given
today's discount curve and `(a, sigma)` it reprices the curve exactly and prices
bond options, caps/floors and European swaptions in closed form, then calibrates
`(a, sigma)` to a swaption grid.

Source: [`nablatensor-quant/.../HullWhiteAnalytic.java`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/HullWhiteAnalytic.java)
· [`HullWhiteCalibration.java`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/HullWhiteCalibration.java)
· example [`HullWhiteCalibrationShowcase.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/HullWhiteCalibrationShowcase.java)

## What it does

```java
HullWhiteAnalytic hw = HullWhiteAnalytic.of(discountCurve, /*a*/ 0.08, /*sigma*/ 0.011);

hw.bondReconstitution(t, T, rt);          // P(t,T | r(t)) = A(t,T) e^{-B(t,T) r(t)}
hw.zeroBondCall(optExpiry, bondMat, K);   // and zeroBondPut
hw.caplet(reset, accrual, strikeRate);    // (1 + K tau) puts on P(T, T+tau)
hw.cap(firstReset, accrual, periods, K);
hw.payerSwaption(expiry, accrual, periods, K);   // Jamshidian; and receiverSwaption
hw.theta(t);                              // the drift term, for a simulation that needs it
```

- **Curve fit is automatic.** `A(t,T)` is built from `P^M(0, .)`, so the initial
  curve is reproduced with no separate `theta(t)` solve.
- **Swaptions by Jamshidian.** A payer swaption is a put on the fixed-coupon
  bond; once the critical short rate `r*` (coupon bond at par) is found by a 1-D
  solve, it decomposes into a portfolio of puts on the individual zero-coupon
  bonds, each priced by the closed form.
- **Ho-Lee** is the `a -> 0` limit and is handled without dividing by `a`.

## Calibration

```java
List<HullWhiteCalibration.SwaptionQuote> quotes =
    HullWhiteCalibration.grid(expiries, tenors, /*accrual*/ 1.0, normalVols);
HullWhiteCalibration.Result r = HullWhiteCalibration.calibrate(discountCurve, quotes, 0.03, 0.02);
r.a();  r.sigma();  r.rmsePrice();
```

Each quote's ATM normal vol becomes a target price via `Bachelier` (feature F2);
the model price is the Jamshidian swaption; the sum of squared price residuals
is minimised by a bounded Nelder-Mead. The analytic swaption is **not**
recordable (a root-find and `N(x)`), so this is the numerical rather than the
adjoint calibration route — the adjoint route runs against the `HullWhite1F`
Monte-Carlo swaption instead.

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.HullWhiteCalibrationShowcase
```

## What the tests pin

- `bondReconstitution(0, T, r0)` equals the input curve's `P(0,T)` to `1e-9`,
  across the whole grid and in the Ho-Lee limit.
- ZCB option put-call parity (`call - put = P(0,S) - K P(0,T)`) to `1e-12`.
- Swaption payer-receiver parity to `1e-9`; ATM payer equals ATM receiver.
- The analytic Jamshidian swaption matches a 400k-path short-rate Monte-Carlo
  within 2%.
- Calibrating a grid generated from a known `(a*, sigma*)` recovers those
  parameters (`a` within 0.01, `sigma` within 5e-4) with a price RMSE below
  `1e-6`.
