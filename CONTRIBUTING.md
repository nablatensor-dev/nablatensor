# Contributing to NablaTensor

Thanks for looking. NablaTensor is Apache-2.0 with **no CLA** — a PR is all it takes.

## Ground rules

- **Every example is a test is a docs page.** A new product or model block ships
  with a runnable `main` in `nablatensor-examples`, a smoke test, and a Markdown
  page under `docs/`.
- **Numbers, not adjectives.** Benchmark and comparison claims come with the
  harness, the seeds and the machine. Report parity to Monte-Carlo noise.
- **The engine is fixed; the seams are where you extend.** If your change needs
  to fork the engine, that is a bug in the seam design — open an issue first.
- **`cpu-jit` stays clean.** The default backend must build and run with no GPU,
  no native library and no incubator flag. Accelerated backends gate themselves
  at runtime through `AadEngine.isAvailable()`.

## Building

Requires JDK 25+ and Maven 3.9+.

```bash
mvn -o -q test        # must be green with no extra flags
mvn -o -q install
```

The `simd` backend needs `--add-modules jdk.incubator.vector`; the
`nablatensor-validate` and `nablatensor-examples` test runs add it so `simd` is
exercised where present, but nothing else depends on it.

## Layout

| module | add here |
|---|---|
| `nablatensor-core` | engine internals + shared CUDA-C tape codegen — rare, discuss first |
| `nablatensor-cpu` / `-jit-cpu` / `-simd` | CPU backend replay paths |
| `nablatensor-vulkan` / `-rocm` / `-cuda` | GPU replay engines (device codegen + dispatch) |
| `nablatensor-tensor` / `-backend-*` | low-level device runtimes — rare, discuss first |
| `nablatensor-quant` | products, model blocks, the `MonteCarlo` driver |
| `nablatensor-validate` | validation / evidence tooling |
| `nablatensor-examples` | a worked demo (+ its docs page + smoke test) |
| `nablatensor-bench` | a reproducible comparison |

A GPU backend must compile with its toolchain absent and gate at runtime via
`AadEngine.isAvailable()` (which must never throw) — `mvn -o test` stays green
on a machine with no GPU.

## Commit / PR

- One logical change per PR. Keep the diff readable.
- Run `mvn -o -q test` before pushing.
- Describe the *why*. Link the issue or the regulation/keyword the change serves.

## Scope

See the [Roadmap](README.md#roadmap) for the MVP / Phase 1 / 2 / 3 breakdown.
MVP-adjacent contributions (more vanilla/exotic payoffs, more model blocks,
tighter validation) are the easiest to land now. Open an issue or a Discussion
if you're unsure whether something fits.
