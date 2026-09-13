"""QuantLib (C++, via the QuantLib Python wheel): MCEuropeanEngine and
MCDiscreteArithmeticAPEngine on the ladder. Greeks are bump-and-revalue on the same
seed, which is the QuantLib way (its MC engines have no adjoint). One thread, as shipped.

  python ql_bench.py european|asian PATHS [greeks]
"""
import sys
import QuantLib as ql
from ladder import run, rss_mb

product, paths = sys.argv[1], int(sys.argv[2])
greeks = len(sys.argv) > 3 and sys.argv[3] == "greeks"
STEPS = 252

today = ql.Date(1, 1, 2026)
ql.Settings.instance().evaluationDate = today
dc = ql.Actual365Fixed()   # Business252 would give a uniform 1/252 grid, and costs 150x in Business252::yearFraction
DAYS = 365
spot, vol, rate = ql.SimpleQuote(100.0), ql.SimpleQuote(0.20), ql.SimpleQuote(0.03)
process = ql.BlackScholesProcess(
    ql.QuoteHandle(spot),
    ql.YieldTermStructureHandle(ql.FlatForward(today, ql.QuoteHandle(rate), dc)),
    ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), ql.QuoteHandle(vol), dc)))


def european(strike, expiry_days):
    return ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, strike),
                            ql.EuropeanExercise(today + expiry_days))


def asian(strike, extra_days):
    fixings = [today + round(i * (DAYS + extra_days) / STEPS) for i in range(1, STEPS + 1)]
    return ql.DiscreteAveragingAsianOption(ql.Average.Arithmetic, 0.0, 0, fixings,
                                           ql.PlainVanillaPayoff(ql.Option.Call, strike),
                                           ql.EuropeanExercise(fixings[-1]))


def engine(seed):
    if product == "european":
        return ql.MCEuropeanEngine(process, "pseudorandom", timeSteps=1, requiredSamples=paths, seed=seed)
    return ql.MCDiscreteArithmeticAPEngine(process, "pseudorandom", requiredSamples=paths, seed=seed)


H_S, H_K, H_V, H_R = 0.01, 0.01, 1e-4, 1e-4
make = european if product == "european" else asian
base, bumpK, bumpT = make(100.0, 0), make(100.0 + H_K, 0), make(100.0, 1)   # dT: the whole 252-fixing grid stretched by one day
if product == "european":
    base, bumpK, bumpT = european(100.0, DAYS), european(100.0 + H_K, DAYS), european(100.0, DAYS + 1)


def value(s, seed):
    eng = engine(seed)
    for o in (base, bumpK, bumpT):
        o.setPricingEngine(eng)
    spot.setValue(s)
    v = base.NPV()
    if not greeks:
        return {"price": v}
    se = base.errorEstimate()
    spot.setValue(s + H_S); d = (base.NPV() - v) / H_S; spot.setValue(s)
    vol.setValue(0.20 + H_V); vg = (base.NPV() - v) / H_V; vol.setValue(0.20)
    rate.setValue(0.03 + H_R); rh = (base.NPV() - v) / H_R; rate.setValue(0.03)
    dk = (bumpK.NPV() - v) / H_K
    dT = (bumpT.NPV() - v) / (1.0 / DAYS)
    return {"price": v, "se": se, "delta": d, "dK": dk, "vega": vg, "rho": rh, "dT": dT}


r = run(f"QuantLib {ql.__version__} {product}", value, paths, greeks,
        {"threads": 1, "greeks": "bump x5, same seed" if greeks else "-"})
print("peak RSS MB", rss_mb())
