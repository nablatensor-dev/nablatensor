"""tf-quant-finance (TensorFlow): GeometricBrownianMotion.sample_paths on the ladder.
Greeks are a tf.GradientTape over spot, strike, vol, rate and maturity. Threads are
TensorFlow's intra-op pool, pinned to 8. `normal_draws` is tff's own way to supply
draws, and the "cached" mode keeps the block per seed the way DRAW_CACHE does.

  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python tff_bench.py european|asian PATHS [greeks] [cached] [xla]
"""
import os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
import tf_quant_finance as tff
from ladder import run, rss_mb

product, paths = sys.argv[1], int(sys.argv[2])
flags = set(sys.argv[3:])
greeks, cached, xla = "greeks" in flags, "cached" in flags, "xla" in flags
THREADS = 8
tf.config.threading.set_intra_op_parallelism_threads(THREADS)
tf.config.threading.set_inter_op_parallelism_threads(1)
STEPS = 1 if product == "european" else 252
D = tf.float64


@tf.function(jit_compile=xla)
def price(spot, strike, vol, rate, T, seed, draws):
    times = tf.linspace(T / STEPS, T, STEPS)
    gbm = tff.models.GeometricBrownianMotion(rate, vol, dtype=D)
    kw = {"normal_draws": draws} if draws is not None else \
         {"random_type": tff.math.random.RandomType.STATELESS, "seed": seed}
    s = gbm.sample_paths(times=times, initial_state=spot, num_samples=paths, **kw)[..., 0]
    avg = s[:, -1] if STEPS == 1 else tf.reduce_mean(s, axis=1)
    payoff = tf.nn.relu(avg - strike) * tf.exp(-rate * T)
    return tf.reduce_mean(payoff), tf.math.reduce_std(payoff) / np.sqrt(paths)


@tf.function(jit_compile=xla)
def price_and_greeks(spot, strike, vol, rate, T, seed, draws):
    with tf.GradientTape() as tape:
        tape.watch([spot, strike, vol, rate, T])
        v, se = price(spot, strike, vol, rate, T, seed, draws)
    g = tape.gradient(v, [spot, strike, vol, rate, T])
    return v, se, g


draw_block = {}


def draws_for(seed):
    if seed not in draw_block:
        draw_block.clear()
        draw_block[seed] = tf.random.stateless_normal([paths, STEPS, 1], seed=[seed, 0], dtype=D)
    return draw_block[seed]


def value(s, seed):
    args = (tf.constant(s, D), tf.constant(100.0, D), tf.constant(0.20, D), tf.constant(0.03, D),
            tf.constant(1.0, D), tf.constant([seed, 0], tf.int32), draws_for(seed) if cached else None)
    if greeks:
        v, se, g = price_and_greeks(*args)
        d, dk, vg, rh, dT = [float(x) for x in g]
        return {"price": float(v), "se": float(se), "delta": d, "dK": dk, "vega": vg, "rho": rh, "dT": dT}
    v, se = price(*args)
    return {"price": float(v), "se": float(se)}


name = f"tf-quant-finance 0.0.1.dev34 / TF {tf.__version__} {product}" + (" xla" if xla else "")
run(name, value, paths, greeks, {"threads": THREADS, "greeks": "GradientTape, 5" if greeks else "-",
                                  "draws": "normal_draws block per seed" if cached else "stateless per call"})
print("peak RSS MB", rss_mb())
