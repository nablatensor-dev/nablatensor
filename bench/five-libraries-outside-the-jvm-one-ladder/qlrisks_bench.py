"""QuantLib-Risks (QuantLib 1.33 built on XAD): the same two MC engines as ql_bench.py,
with the Greeks from one adjoint sweep over the whole C++ valuation. The tape holds
every arithmetic operation of every path, so the path counts are what fits in memory
(about 1.3 KB per European path, 170 KB per 252-step Asian path). One thread.
The price-only rows run with the tape deactivated. No dT: the time grid is dates.

  python qlrisks_bench.py european|asian PATHS [greeks]
"""
import sys
import QuantLib_Risks as ql          # must be imported before xad: it patches Tape.activate
from xad.adj_1st import Real
from xad import value as xv
from ladder import run, rss_mb

product, paths = sys.argv[1], int(sys.argv[2])
greeks = len(sys.argv) > 3 and sys.argv[3] == "greeks"
STEPS, DAYS = 252, 365
today = ql.Date(1, 1, 2026)
ql.Settings.instance().evaluationDate = today
dc = ql.Actual365Fixed()


def value(s, seed):
    with ql.Tape() as tape:
        s0, k0, v0, r0 = Real(s), Real(100.0), Real(0.20), Real(0.03)
        if greeks:
            for x in (s0, k0, v0, r0):
                tape.registerInput(x)
            tape.newRecording()
        else:
            tape.deactivate()
        spot, vol, rate = ql.SimpleQuote(s0), ql.SimpleQuote(v0), ql.SimpleQuote(r0)
        process = ql.BlackScholesProcess(
            ql.QuoteHandle(spot),
            ql.YieldTermStructureHandle(ql.FlatForward(today, ql.QuoteHandle(rate), dc)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), ql.QuoteHandle(vol), dc)))
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, k0)
        if product == "european":
            opt = ql.VanillaOption(payoff, ql.EuropeanExercise(today + DAYS))
            opt.setPricingEngine(ql.MCEuropeanEngine(process, "pseudorandom", timeSteps=1, requiredSamples=paths, seed=seed))
        else:
            fixings = [today + round(i * DAYS / STEPS) for i in range(1, STEPS + 1)]
            opt = ql.DiscreteAveragingAsianOption(ql.Average.Arithmetic, 0.0, 0, fixings, payoff, ql.EuropeanExercise(fixings[-1]))
            opt.setPricingEngine(ql.MCDiscreteArithmeticAPEngine(process, "pseudorandom", requiredSamples=paths, seed=seed))
        v = opt.NPV()
        out = {"price": xv(v), "se": xv(opt.errorEstimate())}
        if greeks:
            tape.registerOutput(v)
            v.derivative = 1.0
            tape.computeAdjoints()
            out.update({"delta": s0.derivative, "dK": k0.derivative, "vega": v0.derivative, "rho": r0.derivative})
        return out


run(f"QuantLib-Risks {ql.__version__} (XAD) {product}", value, paths, greeks,
    {"threads": 1, "greeks": "XAD adjoint, 4 (no dT)" if greeks else "- (tape deactivated)"})
print("peak RSS MB", rss_mb())
