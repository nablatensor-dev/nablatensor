"""LIBOR market model swaption, MatLogica AADC — Mike Giles' benchmark by way of
ad-benchmarks/src/libor_swaption.hpp and bench/lib-comparison-adbench/LiborBench.java.
161 differentiable inputs (1 accrual delta + 80 forward rates L0 + 80 vols lambda),
15 swaptions, 10,000 paths. L0[0] is never stochastically evolved (Giles' own
algorithm, reproduced exactly) but still has a real gradient through the final
discount-back division — checked against a finite-difference bump below.

  python libor_matlogica.py
"""
import numpy as np
import aadc
from bench_timing import time_it

NN, N, SAMPLES = 80, 40, 40
MATURITIES = [4, 4, 4, 8, 8, 8, 20, 20, 20, 28, 28, 28, 40, 40, 40]
SWAPRATES = [.045, .05, .055, .045, .05, .055, .045, .05, .055, .045, .05, .055, .045, .05, .055]
PATHS = 10_000
SEED = 12354
THREADS = 8

pool = aadc.ThreadPool(THREADS)
k = aadc.Functions()
k.start_recording()

delta = aadc.idouble(0.05); delta_a = delta.mark_as_input()
L0 = []
L0_args = []
for i in range(NN):
    x = aadc.idouble(0.05); L0.append(x); L0_args.append(x.mark_as_input())
lam = []
lam_args = []
for i in range(NN):
    x = aadc.idouble(0.20); lam.append(x); lam_args.append(x.mark_as_input())

z_arr = aadc.array(np.zeros((1, SAMPLES)))
z_arg = z_arr.mark_as_input_no_diff()[0]
z_row = z_arr[0]

L = list(L0)
sqrt_delta = aadc.math.sqrt(delta)
for n in range(SAMPLES):
    sqez = sqrt_delta * z_row[n]
    v = delta * 0.0
    for i in range(n + 1, NN):
        lami = lam[i - n - 1]
        con1 = delta * lami
        denom = L[i] * delta + 1.0
        term = con1 * L[i] / denom
        v = v + term
        growth = aadc.math.exp(con1 * v + lami * (sqez - con1 * 0.5))
        L[i] = L[i] * growth

b = delta * 0.0 + 1.0
s = delta * 0.0
btmp = [None] * NN
stmp = [None] * NN
for n in range(N, NN):
    b = b / (L[n] * delta + 1.0)
    s = s + b * delta
    btmp[n] = b
    stmp[n] = s

v = delta * 0.0
for i in range(len(MATURITIES)):
    m = MATURITIES[i] + N - 1
    swapval = btmp[m] + stmp[m] * SWAPRATES[i] - 1.0
    v = v + aadc.math.min(swapval, 0.0) * (-100.0)
for n in range(N):
    v = v / (L[n] * delta + 1.0)

v_out = v.mark_as_output()
k.stop_recording()

INPUT_ARGS = [delta_a] + L0_args + lam_args

rng = np.random.default_rng(SEED)
z = rng.standard_normal((SAMPLES, PATHS))

base_inputs = {delta_a: 0.05}
for a in L0_args:
    base_inputs[a] = 0.05
for a in lam_args:
    base_inputs[a] = 0.20
for i in range(SAMPLES):
    base_inputs[z_arg[i]] = z[i]

grad_request = {v_out: INPUT_ARGS}
primal_request = {v_out: []}


def eval_with(overrides, request):
    inputs = dict(base_inputs)
    inputs.update(overrides)
    return aadc.evaluate(k, request, inputs, pool)


# finite-difference sanity check on delta, L0[0], lambda[0], common random numbers
# (same z draws for bump up/down, so MC noise mostly cancels)
h = 1e-6


def bump_fd(arg, base_value):
    f_up = float(np.mean(eval_with({arg: base_value + h}, primal_request)[0][v_out]))
    f_down = float(np.mean(eval_with({arg: base_value - h}, primal_request)[0][v_out]))
    return (f_up - f_down) / (2 * h)


res0 = aadc.evaluate(k, grad_request, base_inputs, pool)
price = float(np.mean(res0[0][v_out]))

fd_delta = bump_fd(delta_a, 0.05)
fd_l0 = bump_fd(L0_args[0], 0.05)
fd_lambda = bump_fd(lam_args[0], 0.20)
adj_delta = float(np.mean(res0[1][v_out][delta_a]))
adj_l0 = float(np.mean(res0[1][v_out][L0_args[0]]))
adj_lambda = float(np.mean(res0[1][v_out][lam_args[0]]))

print(f"  sanity: d(price)/d(delta)    adjoint={adj_delta:.6f}  finite-diff={fd_delta:.6f}")
print(f"  sanity: d(price)/d(L0_0)     adjoint={adj_l0:.6f}  finite-diff={fd_l0:.6f}")
print(f"  sanity: d(price)/d(lambda_0) adjoint={adj_lambda:.6f}  finite-diff={fd_lambda:.6f}")

primal_ms = time_it(lambda: aadc.evaluate(k, primal_request, base_inputs, pool), warmup=3, iters=10)
grad_ms = time_it(lambda: aadc.evaluate(k, grad_request, base_inputs, pool), warmup=3, iters=10)

sum_l0 = sum(float(np.mean(res0[1][v_out][a])) for a in L0_args)
sum_lambda = sum(float(np.mean(res0[1][v_out][a])) for a in lam_args)

print(f"RESULT bench=LiborSwaption lib=MatLogica threads={THREADS} paths={PATHS} inputs=161 | "
      f"primal {primal_ms:.4f} ms | gradient {grad_ms:.4f} ms | price {price:.6f}")
print(f"  sum(dV/dL0)={sum_l0:.4f} sum(dV/dlambda)={sum_lambda:.4f} dV/ddelta={adj_delta:.4f}")
