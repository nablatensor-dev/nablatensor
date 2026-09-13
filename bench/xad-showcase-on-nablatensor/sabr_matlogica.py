"""SABR (Hagan 2002) surface calibration, MatLogica AADC — same formula and fake
500-iteration optimizer loop as ad-benchmarks/src/sabr.hpp and
bench/lib-comparison-adbench/SabrBench.java. 15 differentiable inputs (5 expiries x
alpha/rho/nu), no Monte Carlo: the kernel is recorded once and evaluated 500 times
with a fixed perturbation schedule between calls, same as the NablaTensor harness.

  python sabr_matlogica.py
"""
import math

import numpy as np
import aadc
from bench_timing import time_it

EXPIRIES = 5
STRIKES = 20
PARAMS = 3 * EXPIRIES
CALIB_ITERS = 500
THREADS = 8

expiries = [1.0, 2.0, 5.0, 10.0, 20.0]
forwards = [0.025, 0.028, 0.030, 0.032, 0.035]
true_alpha = [0.040, 0.038, 0.035, 0.032, 0.030]
true_rho = [-0.15, -0.20, -0.25, -0.28, -0.30]
true_nu = [0.50, 0.45, 0.40, 0.35, 0.30]
init_alpha = [a + 0.005 for a in true_alpha]
init_rho = [r + 0.10 for r in true_rho]
init_nu = [n - 0.05 for n in true_nu]

slices = []  # (F, expiry, beta, strikes[], market_vols[])


def hagan_double(alpha, beta, rho, nu, F, K, expiry):
    one_m_beta = 1.0 - beta
    one_m_beta2 = one_m_beta * one_m_beta
    one_m_beta4 = one_m_beta2 * one_m_beta2
    fk_pow = (F * K) ** (one_m_beta / 2.0)
    fk_pow_full = (F * K) ** one_m_beta
    log_fk = math.log(F / K)
    log_fk2 = log_fk * log_fk
    log_fk4 = log_fk2 * log_fk2
    correction = 1.0 + (one_m_beta2 / 24.0 * alpha * alpha / fk_pow_full
                        + 0.25 * rho * beta * nu * alpha / fk_pow
                        + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu) * expiry
    if abs(F - K) < 1e-7 * F:
        return alpha / (F ** one_m_beta) * correction
    denom_geo = fk_pow * (1.0 + one_m_beta2 / 24.0 * log_fk2 + one_m_beta4 / 1920.0 * log_fk4)
    z = nu / alpha * fk_pow * log_fk
    sqrt_term = math.sqrt(1.0 - 2.0 * rho * z + z * z)
    x_z = math.log((sqrt_term + z - rho) / (1.0 - rho))
    return alpha / denom_geo * (z / x_z) * correction


for e in range(EXPIRIES):
    F, T, beta = forwards[e], expiries[e], 0.5
    k_min, k_max = F * 0.3, F * 2.0
    strikes = [k_min + (k_max - k_min) * i / (STRIKES - 1) for i in range(STRIKES)]
    market_vols = [hagan_double(true_alpha[e], beta, true_rho[e], true_nu[e], F, kk, T) for kk in strikes]
    slices.append((F, T, beta, strikes, market_vols))

max_err = max(
    abs(hagan_double(true_alpha[e], slices[e][2], true_rho[e], true_nu[e], slices[e][0], kk, slices[e][1]) - mv)
    for e in range(EXPIRIES) for kk, mv in zip(slices[e][3], slices[e][4])
)
print(f"  sanity: Hagan @ true params vs synthetic market vols, max abs err = {max_err:.3e}")


def hagan_ad(alpha, beta, rho, nu, F, K, expiry):
    """beta, F, K, expiry are record-time constants (plain Python floats), matching
    ad-benchmarks' templated function where only alpha/rho/nu vary under AD."""
    one_m_beta = 1.0 - beta
    one_m_beta2 = one_m_beta * one_m_beta
    one_m_beta4 = one_m_beta2 * one_m_beta2
    fk_pow = (F * K) ** (one_m_beta / 2.0)
    fk_pow_full = (F * K) ** one_m_beta
    log_fk = math.log(F / K)
    log_fk2 = log_fk * log_fk
    log_fk4 = log_fk2 * log_fk2
    correction = (alpha * alpha / fk_pow_full * (one_m_beta2 / 24.0)
                  + rho * nu * alpha / fk_pow * (0.25 * beta)
                  + nu * nu * rho * rho * (-3.0 / 24.0) + nu * nu * (2.0 / 24.0))
    correction = correction * expiry + 1.0
    if abs(F - K) < 1e-7 * F:
        return alpha / (F ** one_m_beta) * correction
    denom_geo = fk_pow * (1.0 + one_m_beta2 / 24.0 * log_fk2 + one_m_beta4 / 1920.0 * log_fk4)
    z = nu / alpha * fk_pow * log_fk
    sqrt_term = aadc.math.sqrt(1.0 - 2.0 * rho * z + z * z)
    x_z = aadc.math.log((sqrt_term + z - rho) / (1.0 - rho))
    return alpha / denom_geo * (z / x_z) * correction


pool = aadc.ThreadPool(THREADS)
k = aadc.Functions()
k.start_recording()

params = []
param_args = []
for e in range(EXPIRIES):
    a = aadc.idouble(init_alpha[e]); aa = a.mark_as_input()
    rho = aadc.idouble(init_rho[e]); rhoa = rho.mark_as_input()
    nu = aadc.idouble(init_nu[e]); nua = nu.mark_as_input()
    params += [a, rho, nu]
    param_args += [aa, rhoa, nua]

total = None
for e in range(EXPIRIES):
    alpha, rho, nu = params[3 * e], params[3 * e + 1], params[3 * e + 2]
    F, T, beta, strikes, market_vols = slices[e]
    for kk, mv in zip(strikes, market_vols):
        model = hagan_ad(alpha, beta, rho, nu, F, kk, T)
        diff = model - mv
        term = diff * diff
        total = term if total is None else total + term

obj_out = total.mark_as_output()
k.stop_recording()

grad_request = {obj_out: param_args}
primal_request = {obj_out: []}

decay_schedule = []
for it in range(CALIB_ITERS):
    decay = 1.0 / (1.0 + 0.01 * it)
    phase = 0.1 * it
    dp = []
    for p in range(PARAMS):
        freq = 1.0 + 0.3 * p
        scale = 0.0002 if p % 3 == 0 else 0.001
        dp.append(scale * decay * math.cos(phase * freq))
    decay_schedule.append(dp)

init_params = []
for e in range(EXPIRIES):
    init_params += [init_alpha[e], init_rho[e], init_nu[e]]


def eval_at(values, request):
    return aadc.evaluate(k, request, dict(zip(param_args, values)), pool)


# finite-difference sanity check on d(objective)/d(alpha_0)
h = 1e-6
bumped = list(init_params)
bumped[0] += h
f_up = float(np.mean(eval_at(bumped, primal_request)[0][obj_out]))
bumped[0] -= 2 * h
f_down = float(np.mean(eval_at(bumped, primal_request)[0][obj_out]))
fd = (f_up - f_down) / (2 * h)
res0 = eval_at(init_params, grad_request)
adjoint = float(np.mean(res0[1][obj_out][param_args[0]]))
print(f"  sanity: d(objective)/d(alpha_0) adjoint={adjoint:.6f}  finite-diff={fd:.6f}")


def full_loop(request):
    values = list(init_params)
    sink = 0.0
    for it in range(CALIB_ITERS):
        r = eval_at(values, request)
        sink += float(np.mean(r[0][obj_out]))
        for p in range(PARAMS):
            values[p] += decay_schedule[it][p]
    return sink


primal_ms = time_it(lambda: full_loop(primal_request), warmup=3, iters=10)
grad_ms = time_it(lambda: full_loop(grad_request), warmup=3, iters=10)

print(f"RESULT bench=SABRCalib lib=MatLogica threads={THREADS} iterations={CALIB_ITERS} | "
      f"primal(full-loop) {primal_ms:.4f} ms | gradient(full-loop) {grad_ms:.4f} ms | "
      f"objective@init {float(np.mean(res0[0][obj_out])):.6e}")
