# Binomial trees: convergence, early exercise, and tree Greeks

*Keywords: binomial tree java, cox ross rubinstein java, leisen reimer java, american option binomial java, backward induction java, richardson extrapolation option java*

Feature **F11**. Binomial trees and backward induction are a curriculum topic in
their own right, and the one valuation the record-and-replay Monte-Carlo engine
cannot do. They live in a small `nablatensor-lattice` module — plain
`double`, `O(n^2)`, no tape — that exists precisely to cover that material.

Source: [`nablatensor-lattice/`](../../nablatensor-lattice/src/main/java/com/nablatensor/lattice/)
· example [`LatticeConvergenceShowcase.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/LatticeConvergenceShowcase.java)

## What's in the module

| Class | What |
|---|---|
| `BinomialTree` | CRR, Jarrow-Rudd, Leisen-Reimer parameterisations; `price(payoff, schedule)` and `priceVanilla(type, strike, schedule)` by backward induction |
| `LatticePayoff` / `ExerciseSchedule` | node exercise value; European / American / Bermudan (`everyNthStep`) schedules |
| `LatticeGreeks` | delta and gamma read off the first two backward-induction slices (free); vega / rho / theta by rebuild |
| `ConvergenceTable` | price versus step count with a one-step Richardson extrapolation |

## Using it

```java
double amer = BinomialTree.of(spot, rate, dividend, vol, T, 2000, BinomialTree.Method.CRR)
    .priceVanilla(OptionType.PUT, strike, ExerciseSchedule.AMERICAN);

LatticeGreeks g = LatticeGreeks.vanilla(spot, rate, dividend, vol, T, 800,
    BinomialTree.Method.CRR, OptionType.CALL, strike, ExerciseSchedule.EUROPEAN);
```

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.LatticeConvergenceShowcase
```

## What the tests pin

- The CRR European option converges to Black-Scholes, and a one-step Richardson
  extrapolation tightens the estimate.
- Leisen-Reimer converges smoothly (no even/odd oscillation) and is within
  `2e-3` of Black-Scholes by 161 steps.
- Put-call parity holds on the tree.
- The American put (`S = K = 40`, `sigma = 0.20`, `T = 1`, `r = 0.06`) matches
  the high-accuracy benchmark `2.3196` on a fine CRR tree.
- A no-dividend American call equals its European; adding a dividend makes early
  exercise valuable. A Bermudan sits between European and American.
- Tree delta, gamma, vega and rho match the closed-form Greeks.

## Deferred

A Hull-White trinomial short-rate tree (for the no-arbitrage-model calibration
that the curriculum teaches on a trinomial lattice) is a follow-up; the analytic
Hull-White model (feature F6) and its Monte-Carlo step block already cover that
chapter's pricing.
