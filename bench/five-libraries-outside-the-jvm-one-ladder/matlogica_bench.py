"""MatLogica AADC (`pip install aadc`, trial license — see the article): record one
scenario as a scalar computational graph, then let `aadc.evaluate` JIT-compile it to
vectorised, multithreaded machine code and replay it over a batch of random draws, one
element of the batch per Monte-Carlo path. Adjoint Greeks are the same recorded graph
with a reverse sweep requested; no bump. Eight threads via `aadc.ThreadPool(8)`, the
library's own parallelism. As shipped: no seed API of its own, so this harness seeds
`numpy`'s generator per call; "cached" keeps the draw block per seed, the same trick
as tf-quant-finance's `normal_draws` and NablaTensor's `DRAW_CACHE`.

  python matlogica_bench.py european|asian PATHS [greeks] [cached]
"""
import sys
import numpy as np
import aadc
from ladder import run, rss_mb

product, paths = sys.argv[1], int(sys.argv[2])
flags = set(sys.argv[3:])
greeks, cached = "greeks" in flags, "cached" in flags
THREADS = 8
STEPS = 1 if product == "european" else 252
S0, K0, VOL0, RATE0, T0 = 100.0, 100.0, 0.20, 0.03, 1.0

pool = aadc.ThreadPool(THREADS)

kernel = aadc.Functions()
kernel.start_recording()

draws = aadc.array(np.zeros((1, STEPS)))
draws_arg = draws.mark_as_input_no_diff()[0]
draws_row = draws[0]

s = aadc.idouble(S0); s_arg = s.mark_as_input()
k = aadc.idouble(K0); k_arg = k.mark_as_input()
vol = aadc.idouble(VOL0); vol_arg = vol.mark_as_input()
rate = aadc.idouble(RATE0); rate_arg = rate.mark_as_input()
T = aadc.idouble(T0); T_arg = T.mark_as_input()

dt = T / STEPS
sqrt_dt = aadc.math.sqrt(dt)
price = s
running = price - price
for i in range(STEPS):
    drift = (rate - 0.5 * vol * vol) * dt
    diffusion = vol * sqrt_dt * draws_row[i]
    price = price * aadc.math.exp(drift + diffusion)
    running = running + price
avg = running / STEPS if product == "asian" else price
disc = aadc.math.exp(-rate * T)
payoff = aadc.math.max(avg - k, 0.0) * disc
payoff_res = payoff.mark_as_output()

kernel.stop_recording()

request = {payoff_res: [s_arg, k_arg, vol_arg, rate_arg, T_arg] if greeks else []}


draw_block = {}


def draws_for(seed):
    if seed not in draw_block:
        draw_block.clear()
        draw_block[seed] = np.random.default_rng(seed).standard_normal((STEPS, paths))
    return draw_block[seed]


def value(spot, seed):
    z = draws_for(seed) if cached else np.random.default_rng(seed).standard_normal((STEPS, paths))
    inputs = {s_arg: spot, k_arg: K0, vol_arg: VOL0, rate_arg: RATE0, T_arg: T0}
    for i in range(STEPS):
        inputs[draws_arg[i]] = z[i]
    res = aadc.evaluate(kernel, request, inputs, pool)
    out = {"price": float(np.average(res[0][payoff_res]))}
    if greeks:
        g = res[1][payoff_res]
        out.update({
            "delta": float(np.average(g[s_arg])),
            "dK": float(np.average(g[k_arg])),
            "vega": float(np.average(g[vol_arg])),
            "rho": float(np.average(g[rate_arg])),
            "dT": float(np.average(g[T_arg])),
        })
    return out


r = run(f"MatLogica AADC (pip trial) {product}", value, paths, greeks,
        {"threads": THREADS, "greeks": "adjoint, 5" if greeks else "-",
         "draws": "numpy block per seed" if cached else "numpy per-call, regenerated"})
print("peak RSS MB", rss_mb())
