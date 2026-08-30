# Calibrate SABR with an adjoint gradient

*Keywords: sabr calibration java, adjoint gradient calibration, volatility smile fit, L-BFGS option calibration, derivative-free vs gradient calibration*

Derivative-free smile calibration is an overnight batch. Record the objective
once, get its exact gradient from one adjoint sweep per iteration, and it is a
sub-second gradient fit.

`Calibrator` takes a *recording* that reads the parameters by name, builds the
sum of squared residuals against the market, and calls `rec.output(sse)`. It
compiles that to one kernel; every L-BFGS iteration is a `setInput` + one
adjoint sweep, so the gradient cost is one extra sweep regardless of the number
of parameters.

```java
Calibrator.Result r = Calibrator.of(rec -> {
        SDouble alpha = rec.input("alpha", 0.20);
        SDouble rho   = rec.input("rho",   0.0);
        SDouble nu    = rec.input("nu",    0.30);
        SDouble beta  = rec.constant(0.5);
        SDouble sse   = rec.constant(0.0);
        for (int i = 0; i < strikes.length; i++) {
            SDouble d = SabrHagan.blackVol(rec, alpha, beta, rho, nu, F, strikes[i], T)
                                 .sub(targetVol[i]);
            sse = sse.add(d.mul(d));
        }
        rec.output(sse);
      })
      .parameter("alpha", 0.20, 1e-4, 2.0)
      .parameter("rho",   0.0, -0.999, 0.999)
      .parameter("nu",    0.30, 1e-4, 5.0)
      .solve();
```

## Run

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.HestonSabrCalibration
```

## Output

```
SABR calibration (adjoint-gradient L-BFGS)

  target params : alpha=0.2840  rho=-0.3100  nu=0.5700
  recovered     : alpha=0.2840  rho=-0.3100  nu=0.5700
  residual SSE  : 6.355e-26
  iterations    : 56   converged=true   1521.4 ms

  strike   target vol   fitted vol
  0.030    1.511224     1.511224
  0.038    1.403076     1.403076
  0.045    1.329400     1.329400
  0.055    1.246528     1.246528
  0.062    1.199711     1.199711
  0.070    1.154536     1.154536
```

The residual is driven to `~1e-25` and the known parameters are recovered to the
printed precision. `CalibrationTest` asserts this and also checks the tape-level
`SabrHagan.blackVol` against the plain-`double` reference to rounding.

The same machinery calibrates any recorded objective — a Heston characteristic
function, a local-vol surface, a curve — by swapping the model function in the
recording.
