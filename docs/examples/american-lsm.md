# American / Bermudan options by least-squares Monte-Carlo

*Keywords: longstaff schwartz java, least squares monte carlo java, american option monte carlo java, bermudan option pricing java, lsm java, policy optimisation exercise boundary java*

Feature **F1**. The `BermudanOption` shell had the exercise-schedule machinery
and a pluggable continuation value, but no way to *fit* the continuation. This
adds it — by **policy optimisation** rather than a probe replay, so the whole
valuation stays on one recorded tape.

Source: [`nablatensor-quant/.../BermudanLsm.java`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/BermudanLsm.java)
· example [`BermudanLsmShowcase.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/BermudanLsmShowcase.java)

## The method

The continuation value at each exercise date is a low-degree polynomial in
log-moneyness whose coefficients `beta` are chosen to **maximise the price**
under the smoothed exercise rule:

- a sub-optimal exercise policy can only lose value, so the optimised price is a
  valid **lower bound**;
- at the optimum `d(price)/d(beta) = 0`, so by the envelope theorem the market
  Greeks read off the same tape with `beta` held fixed are correct to first
  order — no need to differentiate through the fit.

The coefficient gradient comes from `MultiOutput` (one forward sweep, one reverse
sweep per output), exactly like every other adjoint calibration in the library;
a backtracking gradient ascent drives `beta`.

```java
BermudanLsm.Result r = BermudanLsm.price(
    market, OptionType.PUT, /*exerciseDates*/ 25, /*stepsPerDate*/ 6,
    /*polyDegree*/ 3, /*decisionWidth*/ 0.6, /*scenarios*/ 150_000L, /*seed*/ 42L);

r.price();                 // LSM lower bound
r.europeanFloor();         // hold-to-expiry value
r.earlyExercisePremium();  // the difference
r.greeks().spot();         // delta, from the same reverse sweep
```

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.BermudanLsmShowcase -Dpaths=150000 -Ddates=25
```

## What the tests pin

- The American put (`S = K = 40`, `sigma = 0.20`, `T = 1`, `r = 0.06`) lands in
  `(2.15, 2.36)` — below the Longstaff-Schwartz (2001) finite-difference value
  `2.314` by the grid/smoothing gap, as a lower bound should, with a positive
  early-exercise premium.
- An American call on a non-dividend stock collapses to the European: the
  optimiser drives the policy to "hold to expiry".
- Higher volatility raises the early-exercise premium.
- The put delta comes back in `(-1, 0)` from the same reverse sweep.
