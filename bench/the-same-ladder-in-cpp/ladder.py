"""The spot ladder from the draw-cache article, as a harness any library can plug into.

A `value(spot, seed) -> dict` callable prices (and, for a Greeks pass, differentiates)
one scenario. The ladder is: warm-up on seed 7, then LADDERS ladders on seeds 42, 43,
44; each ladder is the seven spots 98..102 in order; cold = first call of a ladder,
warm = mean of the other six; the reported numbers are the medians over ladders.
Throughput is paths / wall-clock of one call, in million paths per second, whatever
the call had to do to produce the Greeks (an adjoint sweep or six revaluations).
"""
import json, os, statistics, sys, time

SPOTS = [98.0, 99.0, 99.5, 100.0, 100.5, 101.0, 102.0]
LADDERS = int(os.environ.get("LADDERS", "3"))
WARMUP = int(os.environ.get("WARMUP", "4"))


def run(name, value, paths, greeks, extra=None):
    for _ in range(WARMUP):
        value(100.0, 7)
    cold, warm, at100, per_spot = [], [], None, []
    for L in range(LADDERS):
        seed = 42 + L
        rates = []
        for s in SPOTS:
            t0 = time.perf_counter()
            r = value(s, seed)
            dt = time.perf_counter() - t0
            rates.append(paths / dt / 1e6)
            if s == 100.0:
                at100 = r
            if L == 0:
                per_spot.append(r)
        cold.append(rates[0])
        warm.append(statistics.mean(rates[1:]))
        print(f"  ladder {L}: cold {rates[0]:.3f}  warm {[round(x, 3) for x in rates[1:]]}", flush=True)
    c, w = statistics.median(cold), statistics.median(warm)
    out = {"lib": name, "paths": paths, "pass": "value+G" if greeks else "price",
           "cold": round(c, 4), "warm": round(w, 4), "at100": at100, "ladder0": per_spot}
    if extra:
        out.update(extra)
    print("RESULT " + json.dumps(out), flush=True)
    return out


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) // 1024
    return -1
