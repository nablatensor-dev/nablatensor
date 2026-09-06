# Multi-curve (OIS) bootstrap with an adjoint Jacobian

*Keywords: multi curve bootstrap java, ois discounting java, tenor basis java, curve bootstrap jacobian java, zero rate risk java, sofr curve java*

Feature **F5**. Post-LIBOR, discounting moved to an OIS curve and each floating
tenor got its own forecast curve. This builds the stack and — because the whole
bootstrap recursion is recorded against `SDouble` quotes and replayed through a
`MultiOutput` — returns the exact `d(zero rate) / d(quote)` Jacobian from one
adjoint sweep, the transformation a rates desk uses to turn instrument PV01s
into zero-rate bucket risk.

Source: [`nablatensor-quant/.../MultiCurveBootstrap.java`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/MultiCurveBootstrap.java)
· [`CurveSet.java`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/CurveSet.java)
· example [`MultiCurveBootstrapShowcase.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/MultiCurveBootstrapShowcase.java)

## Building the stack

```java
MultiCurveBootstrap.Result r = MultiCurveBootstrap.builder()
    .oisSwap(1, 0.0300).oisSwap(2, 0.0315).oisSwap(3, 0.0325)
    .oisSwap(4, 0.0332).oisSwap(5, 0.0338)
    .forecastSwap("3M", 1, 0.0326).forecastSwap("3M", 2, 0.0342)
    .forecastSwap("3M", 3, 0.0353).forecastSwap("3M", 4, 0.0361)
    .forecastSwap("3M", 5, 0.0368)
    .build().solve();

CurveSet cs = r.curves();
cs.df(4.0);                       // OIS discount factor
cs.forwardRate("3M", 2.0, 3.0);  // simple forward off the 3M curve
cs.parSwapRate("3M", 4);         // 4y multi-curve par rate
double[][] jac = r.jacobian();   // d(zero) / d(quote), block lower-triangular
```

Two stages: the OIS discount curve from OIS deposits and par swaps, then each
forecast curve from instruments whose annuity and float legs discount on the OIS
curve. A forecast zero rate therefore depends on the OIS quotes that move its
discounting — those are the cross-block entries of the Jacobian; an OIS zero
never depends on a forecast quote.

## Scope

Stylised annual construction — fixed and float legs share an annual grid, so a
swap of maturity `N` adds exactly one forecast pillar and every step is closed
form. Sub-annual float frequency, curve interpolation inside the solve, and
cross-currency basis are later refinements; the analytic block Jacobian is a
possible optimisation over the recorded-and-replayed one used here.

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.MultiCurveBootstrapShowcase
```

## What the tests pin

- Every calibrating instrument reprices to its quote (`< 1e-9`).
- Feeding the forecast leg the OIS curve's own multi-curve-consistent par rates
  collapses the forecast curve onto the OIS curve.
- The adjoint Jacobian matches a central bump to `1e-5` relative.
- The Jacobian is block lower-triangular: OIS-zero rows are exactly zero against
  forecast quotes; forecast-zero rows react to the OIS quotes.
