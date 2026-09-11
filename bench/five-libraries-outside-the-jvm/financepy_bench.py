"""financepy (numba): EquityVanillaOption.value_mc_numba_parallel (a numba prange
kernel, so it gets 8 threads) and EquityAsianOption.value_mc_fast_vc_numba (its
"standard MC pricer", parallel=False, so it gets one). Both kernels are antithetic
as shipped; the path count is the draw count they are given. Greeks: bump-and-
revalue on the same seed.

In 1.1.2 every EquityVanillaOption.value_mc* method returns NaN (the wrapper passes
opt_type in the slot where the kernel expects r), so the shipped European row is the
closed form, `value()` + delta/vega/rho/theta. "european-kernel" calls the serial
numba kernel directly, in the argument order the kernel declares — not the shipped
API, and the article says so. (The prange kernel is 4-5x faster on eight threads and
gives a different price on every call with the same seed: numba seeds one thread's
generator, so bumps on it are noise.)

  NUMBA_NUM_THREADS=8 python financepy_bench.py european|european-kernel|asian PATHS [greeks]
"""
import os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
import numba
from financepy.utils.date import Date
from financepy.utils.global_types import OptionTypes
from financepy.market.curves.flat_discount_curve import FlatDiscountCurve as DiscountCurveFlat
from financepy.models.black_scholes_mc import value_mc_numba_only, value_mc_numba_parallel
from financepy.models.black_scholes import BlackScholes
from financepy.products.equity.equity_vanilla_option import EquityVanillaOption
from financepy.products.equity.equity_asian_option import EquityAsianOption
from ladder import run, rss_mb

product, paths = sys.argv[1], int(sys.argv[2])
greeks = len(sys.argv) > 3 and sys.argv[3] == "greeks"
STEPS = 252
today = Date(1, 1, 2026)
expiry = today.add_days(365)
div = DiscountCurveFlat(today, 0.0)


def make(strike, extra_days):
    e = today.add_days(365 + extra_days)
    if product.startswith("european"):
        o = EquityVanillaOption(e, strike, OptionTypes.EUROPEAN_CALL)
        o.t_exp = (e - today) / 365.0
        return o
    return EquityAsianOption(today, e, strike, OptionTypes.EUROPEAN_CALL, STEPS)


def npv(opt, s, vol, r, seed):
    if product == "european":
        return opt.value_mc_numba_parallel(today, s, DiscountCurveFlat(today, r), div, BlackScholes(vol), paths, seed, 0)
    if product == "european-kernel":
        return value_mc_numba_only(s, opt.t_exp, opt.strike_price, r, 0.0, vol, 1, paths, seed, 0)
    return opt.value_mc_fast_vc_numba(today, s, DiscountCurveFlat(today, r), div, BlackScholes(vol), paths, seed, 0.0)


H_S, H_K, H_V, H_R = 0.01, 0.01, 1e-4, 1e-4
base, bumpK, bumpT = make(100.0, 0), make(100.0 + H_K, 0), make(100.0, 1)


def closed_form(s, seed):
    dc, m = DiscountCurveFlat(today, 0.03), BlackScholes(0.20)
    v = base.value(today, s, dc, div, m)
    if not greeks:
        return {"price": v}
    return {"price": v, "delta": base.delta(today, s, dc, div, m), "vega": base.vega(today, s, dc, div, m),
            "rho": base.rho(today, s, dc, div, m), "dT": -base.theta(today, s, dc, div, m)}


def value(s, seed):
    v = npv(base, s, 0.20, 0.03, seed)
    if not greeks:
        return {"price": v}
    d = (npv(base, s + H_S, 0.20, 0.03, seed) - v) / H_S
    vg = (npv(base, s, 0.20 + H_V, 0.03, seed) - v) / H_V
    rh = (npv(base, s, 0.20, 0.03 + H_R, seed) - v) / H_R
    dk = (npv(bumpK, s, 0.20, 0.03, seed) - v) / H_K
    dT = (npv(bumpT, s, 0.20, 0.03, seed) - v) / (1.0 / 365)
    return {"price": v, "delta": d, "dK": dk, "vega": vg, "rho": rh, "dT": dT}


import financepy
threads = 1
if product == "european":   # the shipped MC methods return NaN; time the closed form instead, one valuation = one "path"
    run(f"financepy {financepy.__version__} european closed form (value_mc* return NaN)", closed_form, 1, greeks,
        {"threads": 1, "greeks": "analytic delta, vega, rho, theta" if greeks else "-", "mc_value_mc": npv(base, 100.0, 0.20, 0.03, 42)})
else:
    run(f"financepy {financepy.__version__} {product}", value, paths, greeks,
        {"threads": threads, "greeks": "bump x5, same seed" if greeks else "-", "kernel":
         "value_mc_numba_only called directly (serial, antithetic)" if product == "european-kernel" else "value_mc_fast_vc_numba (antithetic + geometric control variate)"})
print("peak RSS MB", rss_mb())
