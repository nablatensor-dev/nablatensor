"""pfhedge (PyTorch): BrownianStock + EuropeanOption, and a 12-line Asian subclass of
BaseDerivative (pfhedge has no Asian; its derivatives are European, lookback, binary,
variance swap). pfhedge has no rate and no discounting — it is a hedging library —
so mu = r and exp(-rT) are applied by the harness. Greeks are torch autograd through
spot, strike, sigma and mu passed as tensors; dt cannot be a tensor (new_tensor()
detaches it), so there is no dT. Threads: torch intra-op, pinned to 8. Seed:
torch.manual_seed, draws regenerated per call.

  python pfhedge_bench.py european|asian PATHS [greeks]
"""
import math, sys
import torch
import pfhedge
from pfhedge.instruments import BrownianStock, EuropeanOption, BaseDerivative
from ladder import run, rss_mb

product, paths = sys.argv[1], int(sys.argv[2])
greeks = len(sys.argv) > 3 and sys.argv[3] == "greeks"
THREADS = 8
torch.set_num_threads(THREADS)
torch.set_default_dtype(torch.float64)
STEPS = 1 if product == "european" else 252


class ArithmeticAsianCall(BaseDerivative):
    def __init__(self, underlier, strike, maturity):
        super().__init__()
        self.register_underlier("underlier", underlier)
        self.strike, self.maturity = strike, maturity

    def payoff_fn(self):
        return torch.relu(self.ul().spot[:, 1:].mean(dim=1) - self.strike)


def value(s, seed):
    torch.manual_seed(seed)
    spot = torch.tensor(s, requires_grad=greeks)
    K = torch.tensor(100.0, requires_grad=greeks)
    sigma = torch.tensor(0.20, requires_grad=greeks)
    r = torch.tensor(0.03, requires_grad=greeks)
    stock = BrownianStock(sigma=sigma, mu=r, dt=1.0 / STEPS)
    deriv = EuropeanOption(stock, strike=K, maturity=1.0) if STEPS == 1 else ArithmeticAsianCall(stock, K, 1.0)
    deriv.simulate(n_paths=paths, init_state=(spot,))
    payoff = deriv.payoff() * torch.exp(-r * 1.0)
    v = payoff.mean()
    out = {"price": v.item(), "se": (payoff.std() / math.sqrt(paths)).item()}
    if greeks:
        d, dk, vg, rh = torch.autograd.grad(v, [spot, K, sigma, r])
        out.update({"delta": d.item(), "dK": dk.item(), "vega": vg.item(), "rho": rh.item()})
    return out


run(f"pfhedge {pfhedge.__version__} / torch {torch.__version__} {product}", value, paths, greeks,
    {"threads": THREADS, "greeks": "autograd, 4 (no dT)" if greeks else "-", "draws": "torch.manual_seed, regenerated"})
print("peak RSS MB", rss_mb())
