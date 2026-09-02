# `demo/` — narrated jshell sessions

Six **real** jshell sessions aimed at a trading / quant audience. Nothing is
scripted output: every number on screen came out of the machine you run it on.

```bash
mvn -o compile          # once
./demo/greeks-on-gpu.sh  # then pick one
./demo/frtb-curvature-on-cuda.sh
./demo/frtb-full-on-cuda.sh
```

| Demo | What you watch happen |
|---|---|
| [`frtb-full-on-cuda.sh`](frtb-full-on-cuda.sh) | **FRTB full** — the whole standardised approach typed out in **ten stages**: loading market/trades/rulebook, the 89 demo parameter buckets, one adjoint delta and three **full shocked repricings on the fastest GPU backend**, the CVR arithmetic, netting onto shared factor keys, `LOW`/`MEDIUM`/`HIGH` aggregation for every class and measure, DRC, RRAO, the capital bridge and sign-off. Every formula is visible jshell code; only the parameter tables are loaded from the example class. |
| [`frtb-curvature-on-cuda.sh`](frtb-curvature-on-cuda.sh) | The expensive part of FRTB curvature on **the fastest GPU backend**: one adjoint delta, then base / +30% / -30% full Asian-option repricings with common random numbers. The GPU work is timed before it becomes CVR and a curvature charge in a few lines. |
| [`greeks-on-gpu.sh`](greeks-on-gpu.sh) | A down-and-in put written as a plain-Java lambda, recorded once, then **20 M paths priced with price + delta + vega + rho + dV/dK + dV/dT from one reverse sweep** on the Vulkan backend. Then a spot ladder and a crash scenario — same kernel, no rebuild. |
| [`adjoint-vs-bump.sh`](adjoint-vs-bump.sh) | The same five Greeks on an Asian call, computed **two ways**: central bump-and-revalue (`1 + 2×5` price-only replays) versus **one adjoint sweep**. Identical numbers to Monte-Carlo noise; the adjoint run never does more than this one sweep however many inputs you add. |
| [`calibrate-a-smile.sh`](calibrate-a-smile.sh) | A SABR `(alpha, rho, nu)` fit to a six-strike vol smile. The objective is recorded once; **every L-BFGS iteration gets its gradient from one reverse sweep** — no finite differences, no hand-coded Hagan derivative. Recovers the parameters to ~1e-13 in a handful of iterations. |
| [`one-tape-every-backend.sh`](one-tape-every-backend.sh) | ServiceLoader backend discovery, then **one recorded Asian-call tape replayed on `cpu` / `cpu-jit` / `simd` / `vulkan` / `rocm`** — every row cross-checked against the scalar `cpu` oracle (fp64 family to rounding, fp32 GPU to ~5 d.p.) and honestly timed (warm-up first, same seed and tape throughout). |

| Flag | Effect |
|---|---|
| `--fast` | no typing delay — for when you just want the numbers |
| `--cpu` | force the `cpu-jit` engine instead of a GPU one |

## Backend

`greeks-on-gpu.sh`, `adjoint-vs-bump.sh`, `frtb-curvature-on-cuda.sh` and
`frtb-full-on-cuda.sh` pick the fastest **fp32** adjoint engine this machine
has — `cuda`, else `vulkan`, else `rocm`, else `cpu-jit` — and say which in a
dim line under the title. `calibrate-a-smile.sh` is a host-side optimiser over
an analytic formula, so it runs on `cpu-jit` regardless. Override the pick with
`NABLATENSOR_DEMO_ENGINE=cuda` (or `vulkan` / `rocm` / `cpu-jit`), or pass
`--cpu` to force `cpu-jit`.

The two FRTB demos' default workloads are five million and two million paths,
respectively, at 252 fixings.

Needs JDK 24+ for the Class-File API. The scripts ignore `$JAVA_HOME` (often an
older JDK) and use `/opt/zulu26.32.13-ca-jdk26.0.2-linux_x64`; override with
`$NABLATENSOR_JDK`.

The boxes are 76 columns wide, so a 100-column terminal is a good choice.

## `_player.sh`

Shared plumbing: the palette, the box drawing, the typing effect, and a jshell
coprocess driven line by line so narration and results interleave in order. It
is sourced only by the `demoNN.sh` scripts and is never touched by the build or
the test suite.

Two rules that will bite if you write a new demo:

- **Every typed line must be a complete jshell snippet, or end in `(` `,` `{`.**
  jshell auto-appends the missing semicolon, so a fluent builder chain split
  across lines becomes several snippets and the continuation lines fail. Break
  such chains with intermediate `var`s (see the `MonteCarlo.of(...)` builds).
- **Put a long-running call last in its `run` block.** Output only streams once
  every line of the block has been sent.
