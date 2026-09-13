"""Heston stochastic-vol MC, MatLogica AADC — same formula as
ad-benchmarks/src/heston.hpp and bench/lib-comparison-adbench/HestonBench.java.
8 differentiable inputs, 100 Euler steps, 10,000 paths, draws pre-generated once
(AADC has no built-in MC engine or RNG of its own, so the draw block is numpy,
generated once and reused every timed call — the "cached" convention throughout
this project, and the closest match to ad-benchmarks' own pre-generated arrays).

  python heston_matlogica.py
"""
import numpy as np
import aadc
from bench_timing import time_it

STEPS = 100
RHO = -0.7
PATHS = 10_000
SEED = 77777
THREADS = 8

pool = aadc.ThreadPool(THREADS)

k = aadc.Functions()
k.start_recording()

S0 = aadc.idouble(100.0); S0a = S0.mark_as_input()
K = aadc.idouble(105.0); Ka = K.mark_as_input()
T = aadc.idouble(1.0); Ta = T.mark_as_input()
r = aadc.idouble(0.05); ra = r.mark_as_input()
v0 = aadc.idouble(0.04); v0a = v0.mark_as_input()
kappa = aadc.idouble(2.0); kappaa = kappa.mark_as_input()
theta = aadc.idouble(0.04); thetaa = theta.mark_as_input()
xi = aadc.idouble(0.3); xia = xi.mark_as_input()

z1arr = aadc.array(np.zeros((1, STEPS))); z1a = z1arr.mark_as_input_no_diff()[0]; z1row = z1arr[0]
z2arr = aadc.array(np.zeros((1, STEPS))); z2a = z2arr.mark_as_input_no_diff()[0]; z2row = z2arr[0]

dt = T / STEPS
sqrt_dt = aadc.math.sqrt(dt)
sqrt_1m_rho2 = (1.0 - RHO * RHO) ** 0.5

S, v = S0, v0
for i in range(STEPS):
    dW1 = z1row[i]
    dW2 = RHO * z1row[i] + sqrt_1m_rho2 * z2row[i]
    sqrt_v = aadc.math.sqrt(aadc.math.abs(v) + 1e-10)
    S = S + r * S * dt + sqrt_v * S * sqrt_dt * dW1
    v = v + kappa * (theta - v) * dt + xi * sqrt_v * sqrt_dt * dW2

payoff = aadc.math.max(S - K, 0.0) * aadc.math.exp(-r * T)
payoff_out = payoff.mark_as_output()
k.stop_recording()

INPUT_NAMES = ["S0", "K", "T", "r", "v0", "kappa", "theta", "xi"]
INPUT_ARGS = [S0a, Ka, Ta, ra, v0a, kappaa, thetaa, xia]

rng = np.random.default_rng(SEED)
z1 = rng.standard_normal((STEPS, PATHS))
z2 = rng.standard_normal((STEPS, PATHS))

inputs = {S0a: 100.0, Ka: 105.0, Ta: 1.0, ra: 0.05, v0a: 0.04, kappaa: 2.0, thetaa: 0.04, xia: 0.3}
for i in range(STEPS):
    inputs[z1a[i]] = z1[i]
    inputs[z2a[i]] = z2[i]

grad_request = {payoff_out: INPUT_ARGS}
primal_request = {payoff_out: []}

primal_ms = time_it(lambda: aadc.evaluate(k, primal_request, inputs, pool), warmup=3, iters=10)
res = aadc.evaluate(k, grad_request, inputs, pool)  # keep one result for sanity print
grad_ms = time_it(lambda: aadc.evaluate(k, grad_request, inputs, pool), warmup=3, iters=10)

price = float(np.mean(res[0][payoff_out]))
greeks = {name: float(np.mean(res[1][payoff_out][arg])) for name, arg in zip(INPUT_NAMES, INPUT_ARGS)}

print(f"RESULT bench=HestonMC lib=MatLogica threads={THREADS} paths={PATHS} | "
      f"primal {primal_ms:.4f} ms | gradient {grad_ms:.4f} ms | price {price:.6f}")
print("  greeks: " + " ".join(f"{n}={v:.6f}" for n, v in greeks.items()))
