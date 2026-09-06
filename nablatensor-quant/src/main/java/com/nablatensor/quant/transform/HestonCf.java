/*
 * Copyright 2026 The NablaTensor Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nablatensor.quant.transform;

/**
 * Heston characteristic function of the log-return, in the branch-stable
 * "little Heston trap" form (Albrecher et al. 2007) so a principal-branch
 * complex logarithm is safe for long maturities.
 *
 * <pre>{@code
 * dS/S = r dt + sqrt(v) dW1
 * dv   = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho
 * }</pre>
 */
public record HestonCf(double rate, double v0, double kappa, double theta, double xi, double rho)
    implements CharacteristicFunction {

  @Override
  public Complex phi(double u, double t) {
    Complex iu = new Complex(0.0, u);
    Complex xiRhoIu = iu.mul(rho * xi);                       // rho xi i u
    Complex kMinus = xiRhoIu.neg().add(kappa);                // kappa - rho xi i u

    // d = sqrt( (rho xi i u - kappa)^2 + xi^2 (i u + u^2) )
    Complex d = kMinus.mul(kMinus)
        .add(Complex.real(xi * xi).mul(new Complex(u * u, u)))
        .sqrt();

    // little-trap g = (kMinus - d) / (kMinus + d)
    Complex g = kMinus.sub(d).div(kMinus.add(d));

    Complex edt = d.mul(-t).exp();                            // e^{-d t}
    Complex oneMinusGEdt = Complex.ONE.sub(g.mul(edt));

    // C = r i u t + (kappa theta / xi^2) [ (kMinus - d) t - 2 ln( (1 - g e^{-dt}) / (1 - g) ) ]
    Complex lnTerm = oneMinusGEdt.div(Complex.ONE.sub(g)).log();
    Complex cInner = kMinus.sub(d).mul(t).sub(lnTerm.mul(2.0));
    Complex c = iu.mul(rate * t).add(cInner.mul(kappa * theta / (xi * xi)));

    // D = ((kMinus - d) / xi^2) (1 - e^{-dt}) / (1 - g e^{-dt})
    Complex dCoef = kMinus.sub(d).mul(1.0 / (xi * xi))
        .mul(Complex.ONE.sub(edt))
        .div(oneMinusGEdt);

    return c.add(dCoef.mul(v0)).exp();
  }

  @Override
  public double cumulant1(double t) {
    double emkt = Math.exp(-kappa * t);
    return rate * t + (1.0 - emkt) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * t;
  }

  @Override
  public double cumulant2(double t) {
    // A serviceable approximation: integrated-variance mean plus a vol-of-vol term.
    double emkt = Math.exp(-kappa * t);
    double integratedVar = theta * t + (v0 - theta) * (1.0 - emkt) / kappa;
    double volOfVol = xi * xi * t / (kappa * kappa);          // rough scale for the range
    return integratedVar + volOfVol + Math.abs(rho) * xi * integratedVar / kappa;
  }
}
