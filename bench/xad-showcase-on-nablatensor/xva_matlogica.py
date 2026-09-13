"""XVA CVA, MatLogica AADC — same formula as ad-benchmarks/src/xva.hpp and
bench/lib-comparison-adbench/XvaBench.java: 15-swap portfolio, 20 semi-annual time
buckets, 40 differentiable market inputs (20 zero rates, 10 hazard rates, 10 vol
points), 10,000 paths. Keeps the same upstream quirks: the per-bucket CVA discount
factor reads the ORIGINAL (undiffused) rate curve, and the additive rate-shock
reapplies the same per-path draw at every bucket (see the article for what that does
to the CVA value's magnitude).

  python xva_matlogica.py
"""
import numpy as np
import aadc
from bench_timing import time_it

RATE_TENORS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
               10.0, 12.0, 15.0, 17.0, 20.0, 22.0, 25.0, 27.0, 28.0, 30.0]
HAZARD_TENORS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
NUM_RATES, NUM_HAZARD, NUM_VOLS, NUM_RANDOMS = 20, 10, 10, 40
NUM_SWAPS, PAYMENTS_PER_SWAP, TIME_BUCKETS = 15, 20, 20
DT = 0.5
PATHS = 10_000
SEED = 99999
THREADS = 8

portfolio = []  # (notional, fixed_rate, num_payments, start_time, freq, is_payer)
for i in range(NUM_SWAPS):
    portfolio.append((1_000_000.0, 0.020 + 0.001 * i, PAYMENTS_PER_SWAP, 0.0, DT, i % 2 == 0))


def interp_idx_weight(tenors, t):
    """Record-time (plain float) interpolation index/weight — t is always a fixed
    schedule value, never a batch-varying quantity, so this branches in Python."""
    n = len(tenors)
    if t <= tenors[0]:
        return 0, 0.0, True  # clamp low
    if t >= tenors[-1]:
        return n - 1, 0.0, True  # clamp high
    for i in range(n - 1):
        if t < tenors[i + 1]:
            w = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
            return i, w, False
    return n - 2, 1.0, False


def interp(arr, tenors, t):
    idx, w, clamp = interp_idx_weight(tenors, t)
    if clamp:
        return arr[idx]
    return arr[idx] * (1.0 - w) + arr[idx + 1] * w


def discount_factor(rates, t):
    return aadc.math.exp(interp(rates, RATE_TENORS, t) * (-t))


def survival_prob(hazard, t):
    if t <= 0.0:
        return hazard[0] * 0.0 + 1.0
    return aadc.math.exp(interp(hazard, HAZARD_TENORS, t) * (-t))


def price_swap(rates, swap):
    notional, fixed_rate, num_payments, start_time, freq, is_payer = swap
    fixed_leg = rates[0] * 0.0
    float_leg = rates[0] * 0.0
    for i in range(num_payments):
        t = start_time + (i + 1) * freq
        df = discount_factor(rates, t)
        fixed_leg = fixed_leg + df * (notional * fixed_rate * freq)
        t_prev = start_time + i * freq
        df_prev = discount_factor(rates, t_prev)
        fwd_rate = (df_prev / df - 1.0) / freq
        float_leg = float_leg + fwd_rate * df * (notional * freq)
    return (float_leg - fixed_leg) if is_payer else (fixed_leg - float_leg)


pool = aadc.ThreadPool(THREADS)
k = aadc.Functions()
k.start_recording()

rates = []
rate_args = []
for i in range(NUM_RATES):
    x = aadc.idouble(0.02 + 0.001 * i); rates.append(x); rate_args.append(x.mark_as_input())
hazard = []
hazard_args = []
for i in range(NUM_HAZARD):
    x = aadc.idouble(0.01 + 0.001 * i); hazard.append(x); hazard_args.append(x.mark_as_input())
vols = []
vol_args = []
for i in range(NUM_VOLS):
    x = aadc.idouble(0.15 + 0.01 * i); vols.append(x); vol_args.append(x.mark_as_input())

z_arr = aadc.array(np.zeros((1, NUM_RANDOMS)))
z_arg = z_arr.mark_as_input_no_diff()[0]
z_row = z_arr[0]

sqrt_dt = DT ** 0.5
diffused = list(rates)
cva = rates[0] * 0.0
for bucket in range(TIME_BUCKETS):
    t = (bucket + 1) * DT
    diffused = [diffused[i] + vols[i % NUM_VOLS] * sqrt_dt * z_row[i % NUM_RANDOMS] for i in range(NUM_RATES)]

    portfolio_pv = rates[0] * 0.0
    for swap in portfolio:
        notional, fixed_rate, num_payments0, start_time0, freq, is_payer = swap
        payments_elapsed = int(t / freq)
        remaining_payments = num_payments0 - payments_elapsed
        if remaining_payments > 0:
            remaining = (notional, fixed_rate, remaining_payments, t, freq, is_payer)
            portfolio_pv = portfolio_pv + price_swap(diffused, remaining)

    exposure = aadc.math.max(portfolio_pv, 0.0)
    surv_prev = survival_prob(hazard, t - DT)
    surv_curr = survival_prob(hazard, t)
    default_prob = surv_prev - surv_curr
    df = discount_factor(rates, t)  # original (undiffused) curve, matching upstream
    cva = cva + exposure * default_prob * df

cva_out = cva.mark_as_output()
k.stop_recording()

INPUT_ARGS = rate_args + hazard_args + vol_args

rng = np.random.default_rng(SEED)
z = rng.standard_normal((NUM_RANDOMS, PATHS))

inputs = {}
for i, a in enumerate(rate_args):
    inputs[a] = 0.02 + 0.001 * i
for i, a in enumerate(hazard_args):
    inputs[a] = 0.01 + 0.001 * i
for i, a in enumerate(vol_args):
    inputs[a] = 0.15 + 0.01 * i
for i in range(NUM_RANDOMS):
    inputs[z_arg[i]] = z[i]

grad_request = {cva_out: INPUT_ARGS}
primal_request = {cva_out: []}

primal_ms = time_it(lambda: aadc.evaluate(k, primal_request, inputs, pool), warmup=2, iters=7)
res = aadc.evaluate(k, grad_request, inputs, pool)
grad_ms = time_it(lambda: aadc.evaluate(k, grad_request, inputs, pool), warmup=2, iters=7)

price = float(np.mean(res[0][cva_out]))
sum_rate_grad = sum(float(np.mean(res[1][cva_out][a])) for a in rate_args)
sum_hazard_grad = sum(float(np.mean(res[1][cva_out][a])) for a in hazard_args)
sum_vol_grad = sum(float(np.mean(res[1][cva_out][a])) for a in vol_args)

print(f"RESULT bench=XVA-CVA lib=MatLogica threads={THREADS} paths={PATHS} | "
      f"primal {primal_ms:.4f} ms | gradient {grad_ms:.4f} ms | CVA {price:.4e}")
print(f"  sanity: CVA>=0 -> {price >= 0.0} ; sum(dCVA/drate)={sum_rate_grad:.4e} "
      f"sum(dCVA/dhazard)={sum_hazard_grad:.4e} sum(dCVA/dvol)={sum_vol_grad:.4e}")
