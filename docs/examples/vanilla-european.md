# Vanilla European option Greeks in Java

*Keywords: java monte carlo option pricing, adjoint algorithmic differentiation java, java automatic differentiation, black scholes greeks java*

A European call, priced by Monte Carlo, with **delta, vega, rho and strike
sensitivity from one reverse sweep** — no bumping. The closed form is right
there for comparison.

Source: [`nablatensor-examples/.../VanillaEuropeanGreeks.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/VanillaEuropeanGreeks.java)

## The code

```java
EquityMarket market = EquityMarket.atmOneYear();          // S0=K=100, sigma=20%, r=3%, T=1y

try (MonteCarlo mc = MonteCarlo.of(Products.europeanCall())
        .market(market)
        .steps(1)                 // the terminal value is all a European needs
        .fp64()
        .greeks()
        .on("cpu-jit")            // plain Java: no native lib, no incubator flag
        .build()) {

    Pricing p = mc.run(4_000_000, /*seed*/ 42L);
    BlackScholes bs = BlackScholes.of(OptionType.CALL, market);
    // p.price(), p.delta(), p.vega(), p.rho(), p.strikeSensitivity()
}
```

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.VanillaEuropeanGreeks -Dscenarios=4000000
```

## Output

```
engine=cpu-jit  tape=26 nodes  record=2.5 ms  build=17.8 ms
4,000,000 scenarios in 0.025 s  (1.61e+08 scenarios/s)

               adjoint MC    Black-Scholes      abs error
price            9.416744         9.413398       3.35e-03
delta            0.598865         0.598706       1.58e-04
vega            38.686566        38.666812       1.98e-02
rho             50.469718        50.457233       1.25e-02
dV/dK           -0.504697        -0.504572       1.25e-04
```

The adjoint Greeks track Black-Scholes to Monte-Carlo error. The reverse sweep
costs a small constant on top of the price — it does **not** scale with the
number of risk factors, which is the whole point.

## What to change

- **Put instead of call:** `Products.europeanPut()`, `BlackScholes.of(OptionType.PUT, market)`.
- **A different market:** any `new EquityMarket(spot, strike, vol, rate, maturity)`.
- **A path-dependent payoff:** see [Asian Greeks](asian-greeks.md) and [swap the payoff](swap-the-payoff.md).
