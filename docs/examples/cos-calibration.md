# The COS method: characteristic-function pricing and surface calibration

*Keywords: cos method java, fang oosterlee java, heston characteristic function java, heston calibration java, fourier option pricing java, variance gamma pricing java*

Feature **F13**. When a model has a closed-form characteristic function, a
European option is a cosine series in its log-return density — spectral
convergence, and a whole strike slice from one set of `phi` evaluations. This is
the deterministic, fast counterpart to the Monte-Carlo models: the pricing route
a calibration loop wants.

Source: [`nablatensor-quant/.../transform/`](../../nablatensor-quant/src/main/java/com/nablatensor/quant/transform/)
· example [`CosCalibrationShowcase.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/CosCalibrationShowcase.java)

## What's in the package

| Class | What |
|---|---|
| `CharacteristicFunction` | interface — `phi(u, T)` of `X_T = ln(S_T/S_0)` plus the cumulants that set the truncation range |
| `BsmCf` | Black-Scholes — the oracle for the method itself |
| `HestonCf` | Heston, "little Heston trap" branch-stable form |
| `VarianceGammaCf` | Variance-Gamma (the closed-form route the F7 MC step block deferred) |
| `CosMethod` | `price(cf, type, spot, strike, rate, maturity)` — Fang-Oosterlee, puts by parity |
| `HestonCosCalibrator` | fits `(v0, kappa, theta, xi, rho)` to a `(strike, maturity, price)` grid by Nelder-Mead over COS prices |

## Using it

```java
double px = CosMethod.price(new HestonCf(r, v0, kappa, theta, xi, rho),
                            OptionType.CALL, spot, strike, r, maturity);

HestonCosCalibrator.Result fit = HestonCosCalibrator.calibrate(
    spot, r, quotes, new double[] {0.04, 1.0, 0.04, 0.3, -0.3});
fit.v0();  fit.theta();  fit.rmse();
```

## Run it

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.CosCalibrationShowcase
```

## What the tests pin

- COS against Black-Scholes matches the closed form to `1e-7` across strikes,
  and its put-call parity is exact.
- COS Heston agrees with a Heston Monte-Carlo (`HestonModel`) to 3%.
- COS Variance-Gamma recovers Black-Scholes as `nu -> 0`; a symmetric VG fattens
  both tails, a negative-`theta` VG the left tail.
- Calibrating a Heston surface generated from known parameters recovers `v0` and
  `theta` (the well-identified pair) with a price RMSE below `5e-3`.
