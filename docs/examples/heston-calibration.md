# Calibrate Heston to Monte-Carlo target prices

*Keywords: heston calibration java, monte carlo calibration adjoint, levenberg marquardt aad, calibrate stochastic volatility*

Heston has no elementary characteristic function on the primitive op set (no
`sin`/`cos`), so its European price here is Monte-Carlo. That would normally
make gradient calibration hopeless — a bumped Jacobian of an MC price is
swamped by noise. The adjoint fixes it: the residual Jacobian is exact for the
*sampled* estimator, and common random numbers make the residual surface smooth
enough that Levenberg-Marquardt walks straight to the parameters.

## The pieces

- `Calibrator.leastSquares(residualBody)` — the body records the parameters by
  name and returns the named residuals `E[payoff_k] - target_k`.
- `.scenarios(n).seed(s)` — the objective is Monte-Carlo; the same seed every
  iteration (`setInput`, no re-record) keeps the surface smooth.
- Internally each LM step is one `MultiOutput` evaluation: `1 + N` adjoint
  sweeps give the residual values and their full Jacobian in the parameters.

```java
Calibrator.Result r = Calibrator.leastSquares(rec -> {
        SDouble v0  = rec.input("v0",  0.03);
        SDouble xi  = rec.input("xi",  0.35);
        SDouble rho = rec.input("rho", -0.2);
        Map<String, SDouble> resid = new LinkedHashMap<>();
        SDouble[] terminal = hestonPath(rec, v0, xi, rho, kappa, theta);   // full-truncation Euler
        for (int i = 0; i < strikes.length; i++) {
            SDouble call = terminal[0].sub(strikes[i]).max(0.0).mul(disc);
            resid.put("k" + i, call.sub(target[i]));
        }
        return resid;
      })
      .parameter("v0",  0.03, 1e-3, 0.5)
      .parameter("xi",  0.35, 1e-2, 3.0)
      .parameter("rho", -0.2, -0.98, 0.5)
      .scenarios(60_000).seed(8675309L)
      .solve();
```

## Verification

`HestonCalibrationTest` generates the target prices at
`v0 = 0.05, xi = 0.55, rho = -0.6` (kappa, theta held fixed), starts LM from a
perturbed point, and asserts the residual SSE is driven below `1e-6` and every
parameter is recovered — in ~2 seconds. The `SabrHagan` closed-form
calibration ([`sabr-calibration.md`](sabr-calibration.md)) is the deterministic
counterpart; both go through the same `Calibrator`.
