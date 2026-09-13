# Why MatLogica AADC beats `simd` + `DRAW_CACHE` on the Asian Greeks pass

Follow-up investigation to *The same ladder outside the JVM*
(`content/blog/20260913-five-libraries-outside-the-jvm-one-ladder.mdx`), which
reported MatLogica AADC's cached-draws Asian Greeks pass (5.28 Mpath/s) 1.3×
ahead of NablaTensor's `simd` + `DRAW_CACHE` (3.97 Mpath/s) — the one row in
the whole article series where the reference engine is not fastest. This
document is the evidence for *why*, gathered by reproducing both numbers on
the same machine, reading NablaTensor's source, profiling it with JDK Flight
Recorder, and decomposing MatLogica's own timing and diagnostics. It is a
technical research artifact, not blog prose: claims below are labeled
**confirmed** (backed by a specific measurement or a source-code citation),
**ruled out** (a specific test was run and did not support the hypothesis), or
**open** (plausible but not established here).

## Machine and setup

- CPU: AMD Ryzen 7 8845HS (Zen4 mobile), 8 cores / 16 threads (SMT2). L1d 32
  KiB/core, L2 1 MiB/core, L3 16 MiB shared. `avx512f/dq/cd/bw/vl/ifma/vbmi/
  vbmi2/vnni/bitalg/vpopcntdq/bf16` present (`lscpu`).
- NablaTensor: local build against `/home/petr/Projects/highprio/nablatensor`
  (JDK 25, Zulu 25.30.17), `simd` engine, `nablatensor-engine-simd` module.
- MatLogica: `aadc` 1.8.2 (`pip install aadc`, free trial license — see the
  article's licensing section) in an isolated venv.
- CPU only throughout; no GPU backend touched.

## Reproducing the two numbers

Both figures reproduce, with real run-to-run variance. This matters: read the
1.3× in the article as "MatLogica is reliably ahead here," not as a fixed
constant.

**`simd` + `DRAW_CACHE`, Asian value+Greeks, 300k paths, 8 threads** (exact
config that produced the published NablaTensor rows, via
`bench/lib-comparison/CrnLadder.java` against local sources):

```bash
javac --add-modules jdk.incubator.vector -cp "$(cat cp.txt)" -d out CrnLadder.java
java -Xmx16g --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
  -Dengine=simd -Dproduct=asian -Dsteps=252 -Dpaths=300000 -Dgreeks=true -Dthreads=8 -Dnablatensor.crn=on \
  CrnLadder
```

Five repeats, warm Mpath/s: **4.32, 4.34, 4.36, 4.46, 4.47, 4.61** (median
≈4.4). The article's published 3.97 is below this whole range — plausibly a
slower day on shared machine load at measurement time, or simply the low end
of the same distribution; price (5.3010210548890870) and delta
(0.56163041913700080) match the article's published values exactly, so the
configuration is confirmed identical.

**MatLogica AADC, Asian value+Greeks, 200k paths, cached draws**
(`bench/lib-comparison-nonjava/matlogica_bench.py asian 200000 greeks cached`):
repeats gave **4.98, 5.28 (original), 5.35** Mpath/s.

Taking the two distributions at face value, the gap on this machine today is
closer to **1.1–1.3×** than a fixed 1.3×, but MatLogica is ahead in every
single paired comparison. The effect is real, not noise.

## Per-core throughput: essentially tied

Isolating thread count is the first real finding.

`simd`, 1 thread, otherwise identical config:

```bash
java -Xmx16g --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
  -Dengine=simd -Dproduct=asian -Dsteps=252 -Dpaths=300000 -Dgreeks=true -Dthreads=1 -Dnablatensor.crn=on \
  CrnLadder
# => warm 0.86 Mpath/s
```

MatLogica, decomposed with a standalone timing script (below) at
`ThreadPool(1)`, median of 10 trials at 200k paths: **0.816 Mpath/s**.

**Confirmed: single-core throughput is a near-tie, with `simd` nominally ~5%
ahead** (0.86 vs 0.816). Whatever produces the 8-thread gap is not raw
per-core vectorized-kernel speed — it is what happens when eight of those
cores run at once.

## Thread scaling: this is the actual gap

Standalone MatLogica timing script
(`/tmp/…/scratchpad/matlogica_timing.py`, not part of the repo — reproduced
inline below), which also decomposes Python-side overhead:

```python
import time, numpy as np, aadc
# ... records the same GBM/Asian kernel as matlogica_bench.py ...
def run(pool, ntrials=8):
    z = np.random.default_rng(42).standard_normal((STEPS, PATHS))
    for trial in range(ntrials):
        t0 = time.perf_counter()
        inputs = {s_arg: 100.0, k_arg: K0, vol_arg: VOL0, rate_arg: RATE0, T_arg: T0}
        for i in range(STEPS):
            inputs[draws_arg[i]] = z[i]
        t1 = time.perf_counter()
        res = aadc.evaluate(kernel, request, inputs, pool)
        t2 = time.perf_counter()
        g = res[1][payoff_res]; price_v = np.average(res[0][payoff_res]); delta = np.average(g[s_arg])
        t3 = time.perf_counter()
        # record t1-t0 (dict build), t2-t1 (evaluate), t3-t2 (average)
```

Results, 200k paths, median of 10 trials each:

| threads | dict_build | evaluate | average | total | Mpath/s |
|---|--:|--:|--:|--:|--:|
| 1 | 0.084 ms | 244.8 ms | 0.30 ms | 245.2 ms | 0.816 |
| 4 | 0.16 ms | 68.4 ms | 0.32 ms | 68.9 ms | 2.902 |
| 8 | 0.10 ms | 36.4 ms | 0.31 ms | 36.8 ms | 5.431 |

**Confirmed: Python-side overhead is negligible** (dict construction +
`np.average` together are <1.2% of wall time at 8 threads) — the reported
throughput is a clean measurement of the native `aadc.evaluate()` call, not
inflated by omitted bookkeeping.

**Confirmed: MatLogica's 8-thread speedup is 6.65× over 1 thread — 83%
parallel efficiency.** `simd`'s is ≈4.4/0.86 = 5.1× — **64% parallel
efficiency**, using the median of the five repeats above (the spread across
individual pairs of runs gives 58–71%, still clearly below MatLogica's 83%).

**This is the finding: the two engines start from essentially the same
per-core speed, and MatLogica simply scales across 8 cores more efficiently
on this machine, for this tape.** That efficiency gap, not raw kernel speed,
is what flips the ranking.

## What NablaTensor's source and profiler show

Tape size, from a one-off probe (`mc.nodes()`):

```
252-step Asian call, adjoint, threads=8, engine=simd => nodes = 1536
```

`nablatensor-engine-simd/src/main/java/com/nablatensor/engine/simd/VectorReplayF64.java`,
`runRange` (per worker thread, called once per `mc.run(...)` call):

```java
final int n = ops.length;                       // 1536 for this tape
final double[] v = new double[n * BATCH];        // BATCH = 32 (default)
final double[] d = adjoints ? new double[n * BATCH] : null;
final double[] draws = new double[BATCH];
...
for (long done = 0; done < count; done += BATCH) {
  ...
  forward(v, draws, rng, base, seed, crn);
  ...
  Arrays.fill(d, 0.0);                           // whole buffer, every 32-path chunk
  Arrays.fill(d, outRow, outRow + BATCH, 1.0);
  reverse(v, d);
  ...
}
```

`BatchedReplay.java`'s own doc comment confirms the design: *"every sweep
keeps BATCH scenarios side by side per node, so it indexes rows rather than
nodes"* — **one array slot per tape node, for the whole tape, no reuse of
slots whose value is no longer needed.** `v` and `d` are each `1536 × 32 =
49,152` doubles = 384 KiB; 768 KiB per worker thread for the pair; **freshly
allocated on every `runRange()` call** (i.e. every `mc.run(...)`, not once per
`MonteCarlo` build — confirmed by reading the allocation site inside the
method that is called per range/task).

JDK Flight Recorder, same 8-thread config
(`-XX:+FlightRecorder -XX:StartFlightRecording=filename=simd_asian.jfr,settings=profile`),
4-second window covering warmup + the 3-ladder measurement:

```
jdk.ExecutionSample top frames (of 1522 samples):
  568  VectorReplayF64.runRange
  351  VectorPhilox.normals
  263  java.util.Arrays.fill        <- ~17% of all samples
  132  VectorReplayF64.revMul
   80  VectorReplayF64.revAdd
   39  VectorReplayF64.fwdAdd
   28  VectorReplayF64.fwdMul
   21  VectorReplayF64.revExp
```

**Confirmed: `Arrays.fill` — the full-buffer adjoint zero at the top of every
32-path chunk — is the single largest named hotspot after the core sweep
loops**, at roughly 17% of samples. All 8 `aad-simd` worker threads show even
sample counts (179–201, ±6%), so this is not one straggler thread; it is
uniform, structural cost paid by every worker on every chunk.

Per-thread sample and allocation counts do not implicate JIT/GC background
threads: compiler threads (`C1`/`C2 CompilerThread*`) appear in `ThreadCPULoad`
events but not among the sampled hotspots, and are not competing meaningfully
for the profiled window.

### GC: present at warmup, absent at steady state — ruled out as a steady-state cost

`-Xlog:gc` for the *entire* ladder run (warmup + 3 ladders, 25 `mc.run()`
calls):

```
[0.198s] GC(0) Pause Young (Normal) (G1 Evacuation Pause) 177M->137M(760M) 2.043ms
[0.375s] GC(1) Pause Young (Normal) (G1 Evacuation Pause) 393M->137M(760M) 1.283ms
[0.398s] GC(2) Pause Young (Normal) (G1 Evacuation Pause) 489M->134M(760M) 1.150ms
[0.411s] GC(3) Pause Young (Concurrent Start) (G1 Humongous Allocation) 209M->131M(760M) 1.118ms
[0.411s] GC(4) Concurrent Undo Cycle 0.173ms
```

All five GC-related events land in the **first 0.41 seconds** (JVM/class
warmup, before the measured ladder even starts — the harness's own 4-call
warmup loop runs first). **Zero GC events occur during the remaining ~5–8
seconds that cover all 25 measured `mc.run()` calls.** The default G1 young
generation (760 MiB) comfortably absorbs the ~6 MiB of `v`/`d`/`draws` garbage
each `run()` call produces (8 threads × 768 KiB) without collecting.

**Ruled out** (directly tested): forcing a much larger, fixed young generation
to further suppress any possibility of GC —

```bash
java -Xmx16g -Xmn4g ... CrnLadder   # vs default (~760M young gen)
# baseline: 4.47 Mpath/s warm   |   -Xmn4g: 3.95 Mpath/s warm
```

— did not improve throughput (if anything, slightly worse, within
run-to-run noise). **GC pause overhead and allocation-triggered collection
are not the scaling bottleneck.** The `Arrays.fill` cost visible in profiling
is real CPU time spent zeroing memory, but it is not GC time.

### Cache capacity: checked, does not look like the bottleneck

Per-thread `v`+`d` = 768 KiB fits inside this CPU's 1 MiB per-core L2 with
headroom. Aggregate across 8 threads (≈6.1 MiB) fits well inside the 16 MiB
shared L3. `SimdSupport.java`'s own doc comment flags exactly this kind of
capacity tradeoff as a real concern *for other batch-size/node-count
combinations* ("twelve workers turn [1.5 MB/worker at BATCH=64] into 18 MB
against a 12 MB L3"), but the numbers for *this* 1536-node tape at the
default `BATCH=32` do not put the working set near either cache's capacity
limit. **This specific hypothesis does not hold up numerically for this
configuration** — noted so it isn't reached for reflexively as "the" answer.

### AVX-512 width: confirmed active on the NablaTensor side, inconclusive on MatLogica's

```java
VectorSpecies<Double> s = DoubleVector.SPECIES_PREFERRED;
// => Species[double, 8, S_512_BIT]  (8 lanes, 512-bit)
```

**Confirmed:** the JVM Vector API is using full 512-bit (8-lane) `double`
vectors on this machine — `simd` is not stuck on 256-bit AVX2. A vector-width
disadvantage for NablaTensor is ruled out.

MatLogica's actual runtime vector width could not be determined. Its shipped
Python extension
(`aadc/_aadc_core.cpython-312-x86_64-linux-gnu.so`, 13.9 MB) was disassembled
looking for `zmm`/`ymm` register mnemonics as a coarse signal:

```
zmm register mnemonics: 182,640
ymm register mnemonics:  50,516
```

**This is a red herring, not evidence**, and is reported here specifically so
it is not mistaken for a finding: the `zmm`-heavy code in this `.so` is a
statically-linked OpenSSL build (`ossl_aes_gcm_*_avx512`,
`ossl_rsaz_avx512ifma_*`, etc. — TLS/crypto for the license-server
connection), not AADC's numerical kernel. The actual machine code for a
*specific recorded tape* is JIT-compiled into anonymous executable memory at
`kernel.stop_recording()` time and does not exist in any file on disk to
disassemble. **This line of inquiry is genuinely inconclusive** — AADC is
closed-source and its runtime code generation is not inspectable with the
tools used here.

### The one number that plausibly explains the scaling gap: work-array size

MatLogica exposes its own compiled-kernel diagnostics
(`kernel.recording_stats(name)`, present but commented-out in MatLogica's own
published example code — it works when called):

```
AsianCall Work array size   : 275
AsianCall Stack size        : 1522
AsianCall Func + WS mem use : 130604 bytes
AsianCall Num Ops           : 8106
AsianCall Num Rn Variables  : 264
```

**Confirmed, from the vendor's own diagnostic API:** AADC's compiler reduces
the recorded tape to **275 simultaneously-live work-array slots** — i.e. it
performs liveness-based slot reuse, keeping only values that are still needed
resident, discarding the rest. NablaTensor's `simd` engine, per its own source
and design-comment ("indexes rows rather than nodes"), keeps **one array slot
per tape node for the tape's entire lifetime — 1536 slots**, with no reuse of
slots whose value is dead. Per scenario, that is roughly `1536 × 8 bytes =
12,288 bytes` for `v` alone (24,576 bytes counting `d`) against MatLogica's
`275 × 8 = 2,200 bytes` work array — **roughly an order of magnitude more
resident memory per scenario for NablaTensor's approach**, before even
counting the identically-sized adjoint buffer NablaTensor allocates
separately (MatLogica's stats report one combined work-array figure, with no
second buffer of comparable size broken out, suggesting its reverse sweep
reuses the same slots rather than requiring a parallel dense buffer — this
part is *inferred* from the stats, not directly confirmed, since AADC's
internal reverse-mode bookkeeping is not otherwise inspectable).

**This is the most evidence-backed candidate explanation for the scaling
gap**, though it stops short of full confirmation: a ~10× smaller
per-scenario working set means that when 8 threads run at once, MatLogica's
aggregate memory traffic for reading/writing intermediate state is
correspondingly smaller — plausibly enough smaller to stay clear of whatever
shared resource (memory bandwidth, cache-coherency traffic, or something else
not measured here) makes `simd`'s 8-thread case fall short of linear scaling
while MatLogica's does not. This is **stated as a hypothesis consistent with
the evidence, not a proven mechanism** — see "What remains unknown."

## What remains unknown

- **Hardware performance counters were not available.** `perf stat` is
  blocked on this machine (`perf_event_paranoid = 4`, requires elevated
  privileges not requested here). No cache-miss or memory-bandwidth counters
  were gathered for either engine. The work-array-size argument above is
  therefore an inference from a size difference, not a direct bandwidth
  measurement.
- **AADC's internal reverse-mode/threading implementation is closed-source.**
  Its actual JIT-generated vector width, whether/how it re-zeros adjoint
  state, and exactly how `ThreadPool(n)` partitions work across the batch are
  not inspectable beyond what `recording_stats()` reports.
- **The exact 1.3× ratio in the article is not a fixed constant.** Repeated
  measurements on this machine today put both engines a bit higher than
  their published figures, with the gap ranging roughly 1.1–1.3× depending on
  the pair of runs compared. Treat the article's number as representative of
  a real, reproducible, but noisy effect — not a precise multiplier.
- **Why single-core throughput is a near-tie despite the ~10× work-array size
  difference was not explained.** If a smaller working set matters at 8
  threads, it is reasonable to ask why it does not also give MatLogica a
  clear single-core edge; it doesn't, measurably. This asymmetry (structural
  difference visible, but only showing up under multithreading) is real in
  the data and unresolved here — possibly single-core throughput is bound by
  something else entirely (raw op count/dispatch — MatLogica's tape has 8106
  ops against NablaTensor's 1536-node structure, a very different unit of
  work — or instruction-level parallelism inside each kernel — that neither
  engine's working-set size determines).

## Reproduction commands (everything run for this document)

```bash
# --- NablaTensor simd + DRAW_CACHE, Asian, 8 threads ---
cd bench/lib-comparison
mvn -q -o dependency:build-classpath -Dmdep.outputFile=cp.txt
javac --add-modules jdk.incubator.vector -cp "$(cat cp.txt)" -d out CrnLadder.java
java -Xmx16g --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
  -Dengine=simd -Dproduct=asian -Dsteps=252 -Dpaths=300000 -Dgreeks=true -Dthreads=8 -Dnablatensor.crn=on CrnLadder

# --- same, 1 thread, for per-core comparison ---
java -Xmx16g --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
  -Dengine=simd -Dproduct=asian -Dsteps=252 -Dpaths=300000 -Dgreeks=true -Dthreads=1 -Dnablatensor.crn=on CrnLadder

# --- JFR profile ---
java -Xmx16g --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
  -XX:+FlightRecorder -XX:StartFlightRecording=filename=simd_asian.jfr,settings=profile \
  -Dengine=simd -Dproduct=asian -Dsteps=252 -Dpaths=300000 -Dgreeks=true -Dthreads=8 -Dnablatensor.crn=on CrnLadder
jfr print --events jdk.ExecutionSample simd_asian.jfr
jfr print --events jdk.GarbageCollection simd_asian.jfr
jfr print --events jdk.ThreadCPULoad simd_asian.jfr

# --- GC log for the full run ---
java -Xmx16g --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" -Xlog:gc=info:stdout:time,uptime \
  -Dengine=simd -Dproduct=asian -Dsteps=252 -Dpaths=300000 -Dgreeks=true -Dthreads=8 -Dnablatensor.crn=on CrnLadder

# --- large young gen control (GC hypothesis test) ---
java -Xmx16g -Xmn4g --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
  -Dengine=simd -Dproduct=asian -Dsteps=252 -Dpaths=300000 -Dgreeks=true -Dthreads=8 -Dnablatensor.crn=on CrnLadder

# --- tape node count probe ---
# NodesProbe.java: builds the same MonteCarlo config and prints mc.nodes() => 1536

# --- Vector API species probe ---
# VecProbe.java: prints DoubleVector.SPECIES_PREFERRED => Species[double, 8, S_512_BIT]

# --- MatLogica: reproduce the published number ---
cd bench/lib-comparison-nonjava
python matlogica_bench.py asian 200000 greeks cached      # repeat a few times

# --- MatLogica: thread scaling + overhead decomposition ---
# matlogica_timing.py (inline above): ThreadPool(1)/(4)/(8), times dict-build / evaluate / average separately

# --- MatLogica: compiled-kernel diagnostics ---
# call kernel.recording_stats("AsianCall") right after kernel.stop_recording()

# --- native library inspection (inconclusive, see write-up) ---
objdump -d aadc/_aadc_core.cpython-312-x86_64-linux-gnu.so | grep -oE "zmm[0-9]+" | wc -l
strings aadc/_aadc_core.cpython-312-x86_64-linux-gnu.so | grep -i avx512

# --- attempted, blocked by environment ---
perf stat -e task-clock,cycles,instructions,cache-misses,cache-references -- java ... CrnLadder
# => "Access to performance monitoring... limited" (perf_event_paranoid=4)
```

# Part 2 — perf counters, and an attempt to close the gap

Follow-up to the above, done after `perf_event_paranoid=4` was confirmed to be this
machine's Ubuntu-24.04 kernel default (no local sysctl override) rather than something
misconfigured. Rather than lower it — that permanently weakens a real hardening
setting, since unprivileged access to hardware performance counters is itself a known
side-channel surface — every `perf` run below uses `sudo perf stat` per command, which
needs no sysctl change and leaves the machine's posture untouched afterward.

## Part 2a — what the counters actually show

Both binaries were run with `perf stat -D <delay-ms>` (`--delay`), which starts
counting only after the given delay, and with `-Dladders=40` on the Java side /
`LADDERS=40` on the Python side to stretch the measured window to ~20–25 seconds so
the delay cleanly skips JVM class-loading/JIT warmup and Python/aadc import, without
needing to attach mid-run. Events: `task-clock,cycles,instructions,cache-references,
cache-misses`. AMD's PMU on this CPU does not expose `LLC-loads`, `LLC-load-misses` or
`stalled-cycles-backend` (`perf stat` reports them `<not supported>`), so those three
from the original request are dropped; the four that are supported are sufficient to
test the hypothesis.

| | `simd` + `DRAW_CACHE`, 8 threads | MatLogica AADC, cached, 8 threads |
|---|--:|--:|
| task-clock (ms, aggregate) | 151,644.43 | 97,060.99 |
| cycles | 664,998,662,500 | 426,212,208,171 |
| instructions | 1,269,408,405,525 | 818,740,180,722 |
| cache-references | 121,649,028,829 | 78,780,884,586 |
| cache-misses | 5,779,175,768 | 539,457,678 |
| IPC | 1.909 | 1.921 |
| cache-miss rate | 4.751% | 0.685% |

IPC is a near-tie (consistent with Part 1's per-core finding), but the **cache-miss
rate is 6.9× higher for `simd`.** Raw totals are not directly comparable because the
two runs process different amounts of work (300k paths × 252 steps × 284 calls for
`simd`, 200k × 252 × 284 for MatLogica — same `ladders`/`warmup` call count, different
path count), so both series were also normalized per "path-step" (one node's worth of
work for one path for one of the 252 timesteps):

| per path-step | `simd` | MatLogica |
|---|--:|--:|
| cache-references | 5.666 | 5.504 |
| cache-misses | 0.2692 | 0.0377 |
| cycles | 30.97 | 29.78 |

Cache-*reference* volume per unit of work is nearly identical (5.67 vs 5.50 — both
engines touch memory about as often per path-step, which makes sense, they're doing
the same shape of arithmetic). Cache-*miss* volume per unit of work is **7.14× higher
for `simd`.** Cycles per unit of work are close (30.97 vs 29.78, `simd` only 4% worse)
despite that miss-rate gap — consistent with Part 1's single-core near-tie: out-of-order
execution and independent work across the other 7 threads hides most of an individual
miss's latency at the per-thread level. The gap Part 1 found only shows up in
*aggregate throughput at 8 threads*, which is exactly where a shared, bandwidth-limited
resource (the path from L2/L3 to memory, or L3 itself under 8-way concurrent pressure)
would be expected to bite even when no single thread's own critical path is much
longer.

**Confirmed, with hardware counters this time:** `simd`'s larger per-scenario working
set (Part 1's 1536-slot `v`+`d` versus MatLogica's 275-slot work array) produces
measurably more cache misses per unit of work — 7.14× more — which is the strongest
evidence yet for the memory-traffic explanation. **Still open:** the counters show
*more misses*, not directly *more bytes moved over the memory bus* or *contention on a
shared resource* — this machine's PMU does not expose the off-core/DRAM-bandwidth
events that would nail that last link down, so "more misses under concurrent load
plausibly explains worse 8-thread scaling" remains the best-supported story rather
than a fully closed loop.

## Part 2b — attempting the fix: liveness-based slot reuse in `simd`

Given Part 1 and 2a both point at working-set size, the natural next question is
whether compacting NablaTensor's own `v` array (the forward-value buffer;
`nablatensor-engine-simd/…/VectorReplayF64.java` and `BatchedReplay.java` in
`/home/petr/Projects/highprio/nablatensor`, module `nablatensor-engine-simd`) the same
way MatLogica's compiler apparently does closes any of the gap. This section is a real
attempt, not a thought experiment — the code below exists as an uncommitted diff on
that repo's `main` branch (see "State left behind" at the end of this section).

### The idea

`BatchedReplay` currently gives every tape node its own permanent row in both `v`
(forward values) and `d` (adjoints) — `new double[ops.length * BATCH]` for each,
indexed by `node * BATCH`, for the tape's entire lifetime. A node's forward value is
only needed, though, from the step it's computed until the *last* step — forward or
reverse — that reads it. Once nothing can read it again, its row can be handed to a
later node. This is exactly the classic linear-scan register-allocation problem
(Poletto & Sarkar) applied to array rows instead of CPU registers, and it is
well-defined and checkable: compute, once per tape (not per replay), a
`node → compacted slot` mapping, and route every `v[]` read/write in `forward()`/
`reverse()` through it.

**Scope, deliberately narrowed per the task's own guidance:** only `v` is compacted.
`d` (the adjoint array) accumulates via `+=` from every predecessor that propagates
into a node, in whatever order the reverse sweep visits them; getting *its* live
ranges right needs a separate accumulation-aware analysis this pass does not attempt.
`d` keeps one row per node, unchanged, exactly as before.

**The liveness rule that makes this correctness-critical, not just an optimization:**
a node's forward value can be read again during the *reverse* sweep, not only by
later forward consumers. Reading every `rev*` method in `VectorReplayF64.java` gives
the exact rule, opcode by opcode: `EXP`, `DIV` and `SQRT` read their own forward value
back (`v[row]`, e.g. `d(sqrt(x))/dx = 0.5/sqrt(x)`, and the result is already sitting in
`v[row]`); `MUL`, `LOG`, `ABS`, `MAX` and `MIN` read operand A's forward value; `MUL`,
`DIV`, `MAX` and `MIN` read operand B's. `ADD`, `SUB` and `NEG` read no forward value
at all during reverse — their adjoint formulas only touch `d[]`. Treating the full
2-phase schedule as one timeline (forward steps `0..n-1` in node order, then reverse
steps `n..2n-1` visiting node `n-1` first and node `0` last, so node `i`'s reverse step
is at position `2n-1-i`), every node's last-use time is the latest of: the last forward
consumer that needs it as an operand, its own reverse step if its opcode self-reads
(`EXP`/`DIV`/`SQRT`), or the reverse step of whichever consumer reads it as an operand
under the list above. A standard linear-scan pass over nodes in birth order — expire
any row whose occupant's last-use has passed, then take the lowest free row or open a
new one — then gives the minimum possible row count for that constraint set (interval
graphs are perfect graphs; greedy leftmost-fit on sorted intervals is optimal, not
just a heuristic, for this problem).

Implementation: `BatchedReplay.computeVSlots(FlatTape, boolean withAdjoints)` computes
the mapping once in the constructor (the `withAdjoints` flag comes from the engine's
own fixed `options.adjoints()` — a price-only engine instance gets pure forward-only
liveness, no reverse-time extension at all, since `reverse()` is never called for it).
New fields `vRow`, `vRowA`, `vRowB`, `vOutRow`, `vSlotCount` sit alongside the
existing, *unchanged* `rowA`/`rowB`/`inputRow`/`outRow` (still node-indexed, still used
for every `d[]` access). `VectorReplayF64.forward()` now writes/reads `v[]` purely
through the compacted rows. `reverse()` carries **two** sets of row variables per node
— `va`/`vb`/`vrow` (compacted, for the handful of `v[]` reads each opcode's adjoint
formula actually needs) and the original `a`/`b`/`row` (node-indexed, for every `d[]`
read/write, untouched) — because after compaction the two arrays are no longer
addressed by the same index. Eight of the eleven arithmetic `rev*` helper methods'
*bodies* are byte-for-byte unchanged (`revAdd`, `revSub`, `revNeg` never touched `v[]`
to begin with); the other eight — `revMul`, `revDiv`, `revExp`, `revLog`, `revSqrt`,
`revAbs`, `revMax`, `revMin` — gained the extra `v`-row parameters their specific
formula needs, per the opcode list two paragraphs up.

One hard constraint from this codebase's own history shaped the implementation: a test
already in the repo, `nablatensor-examples/…/validate/VectorSweepSplitTest.java`,
pins `forward` and `reverse` as separate methods from `runRange` and puts a 512-byte
bytecode tripwire on `runRange` itself, because folding the sweeps back into the driver
previously caused roughly one JVM in four to crash the C2 compiler outright on a
late-inlined `lanewise(EXP)` call (the class's own doc comment tells the story). The
new fields are computed once in the constructor, not in `runRange`, and every change to
`forward`/`reverse` stays inside those methods — `runRange` itself only gained a
smaller array size and one renamed variable (`outRow` → `vOutRow` for the one line that
reads the final price out of `v[]`). **Confirmed:** `mvn -o -pl nablatensor-examples
test -Dtest=VectorSweepSplitTest` still passes after the change — the tripwire held.

### Correctness verification

Built and installed locally as `0.1.0-SNAPSHOT` (`mvn -pl nablatensor-engine-simd -am
install -DskipTests`, offline), with `bench/lib-comparison/pom.xml` temporarily
repointed at that SNAPSHOT to compile and run `CrnLadder` against it, then restored
to the published `0.1.0` afterward — that pom is back to exactly what it was; `git
status` on `nablatensor-web` shows nothing outstanding from this section.

Price and every Greek were compared, bit-for-bit, between the unmodified engine
(`git stash` before rebuilding) and the modified one, for every combination actually
exercised in the article's tables:

| product | pass | threads | unmodified | modified | match |
|---|---|--:|---|---|---|
| Asian, 300k | value+Greeks | 8 | price 5.3010210548890870, delta 0.56163041913700080 | identical | **bit-exact** |
| Asian, 300k | price-only | 8 | price 5.3010210548890870 | identical | **bit-exact** |
| European, 20M | value+Greeks | 8 | price 9.4122353017758250, delta 0.59869915176805650 | identical | **bit-exact** |
| European, 20M | price-only | 8 | price 9.4122353017758250 | identical | **bit-exact** |
| Asian, 300k | value+Greeks | 1 | price 5.3010210548890770, delta 0.56163041913699730 | identical | **bit-exact** |

(The 1-thread and 8-thread Asian prices differ from each other in the last couple of
ULPs — `…890870` vs `…890770` — in *both* the unmodified and modified engine, identically;
that's floating-point summation order depending on how paths are partitioned across
threads, a pre-existing property of this engine unrelated to this change, and it was
checked to make sure it wasn't mistaken for a bug.)

**Confirmed: the change is correct**, across both products, both passes, and two
thread counts, verified bit-for-bit against the unmodified engine rather than against
a self-consistency check.

### What it actually achieved, and what it cost

The liveness analysis, run on the 1536-node Asian value+Greeks tape via a small
reflection probe reading the engine's own `vSlotCount` field, compacts `v` to **773
rows — 50.3% of 1536, roughly a 2× reduction**, not MatLogica's apparent ~5.6× (275 of
what would be a comparably-sized tape). The reason is structural, not a shortfall in
the analysis: this tape's `EXP` and `SQRT` nodes (one exponential per GBM step, plus
the `sqrt(dt)` term) are exactly the opcodes whose forward value must survive all the
way to *their own* reverse step — the very last thing to happen to a node born early
in a long tape — so a large fraction of nodes end up with last-use times deep in
reverse-time regardless of how well anything else compacts. Getting closer to
MatLogica's ratio on this tape would need restructuring what gets stored, not just
reusing rows more cleverly — e.g. caching just the scalar `e^x` needed for `EXP`'s own
derivative in something smaller than a full `v`-row, rather than keeping the row alive
for a thousand-plus more forward steps and the matching reverse step.

Benchmarked (`-Dengine=simd -Dproduct=asian -Dsteps=252 -Dpaths=300000 -Dgreeks=true
-Dnablatensor.crn=on`, `cpu-jit`/`simd` warm-of-6 medians as elsewhere in this
document):

| | 1 thread | 8 threads |
|---|--:|--:|
| unmodified (Part 1) | 0.86 Mpath/s | 4.32–4.61 Mpath/s (median ≈4.4) |
| modified | 0.74–0.76 Mpath/s | 3.75–4.08 Mpath/s (median ≈3.8) |

**This is a regression, not a win — roughly 12–15% slower at both thread counts.**
`perf stat` on the same steady-state-windowed 8-thread config explains why:

| | unmodified | modified | change |
|---|--:|--:|--:|
| cache-references | 121,649,028,829 | 89,468,610,635 | −26.5% |
| cache-misses | 5,779,175,768 | 3,803,672,064 | **−34.2%** |
| cycles | 664,998,662,500 | 743,153,845,921 | **+11.8%** |
| instructions | 1,269,408,405,525 | 1,234,048,495,653 | −2.8% |
| IPC | 1.909 | 1.661 | −13.0% |

**Confirmed: the memory-traffic hypothesis was mechanically correct** — cache misses
dropped 34%, exactly the direction and rough magnitude the smaller working set
predicts. **Confirmed, and the actual explanation for the regression: the indirection
needed to *achieve* that reduction costs more than it saves at this batch size.** The
unmodified code's per-node row was `i * BATCH` — a loop-counter multiply the JIT
reduces to a free increment, no memory access, no dependency on anything but the loop
variable. The compacted version replaces that with a load from `vRow[i]` (and, for
every binary/unary op, `vRowA[i]`/`vRowB[i]` as well) *before* the row it names can be
used to address `v[]` — a genuine load-to-use dependency chain in the hot loop that
did not exist before, on top of the array read itself. IPC dropping from 1.91 to 1.66
is the signature of exactly that: fewer instructions retired per cycle because more of
them are now waiting on a dependent load rather than issuing independently. A JFR
profile of the modified engine (`jdk.ExecutionSample`, `ladders=20`) shows
`Arrays.fill` (the `d`-array zeroing this document's Part 1 flagged as ~17% of
samples) down to **10.5% of samples (587/5600)** — not because zeroing `d` got
cheaper (`d` is untouched, same size, same cost), but because the denominator grew:
everything else, `revSqrt` most visibly, got more expensive, diluting `Arrays.fill`'s
share of the total.

**Net verdict: a genuine, correctness-verified attempt that made the target metric
worse.** The theory that motivated it is now better-evidenced than before this
section (cache misses really did fall, by the predicted mechanism), but the specific
implementation — row-indirection through fresh per-node arrays — trades memory
bandwidth for dependent-load latency at a rate that loses on this CPU, at `BATCH=32`,
on this tape shape. Two directions that might change that trade, neither attempted
here: (a) restructure self-reading opcodes (`EXP`/`DIV`/`SQRT`) to stash just the
scalar value their own derivative needs in a small dedicated buffer instead of
extending a whole row's lifetime through the entire rest of the tape, which would both
shrink `vSlotCount` well below 773 and remove some of the reverse-time pressure driving
the indirection's cost; (b) encode the compacted index differently — e.g. sorting nodes
so that many consecutive tape positions share the same slot number, turning some of
the `vRow[i]` array loads back into cheap arithmetic on contiguous runs — rather than
an arbitrary per-node mapping. Both are real engineering, not quick follow-ups, and
neither was attempted in the time available for this document.

### State left behind

`/home/petr/Projects/highprio/nablatensor` is on `main`, with exactly two files
modified and **uncommitted** — `git status --short`:

```
 M nablatensor-engine-simd/src/main/java/com/nablatensor/engine/simd/BatchedReplay.java
 M nablatensor-engine-simd/src/main/java/com/nablatensor/engine/simd/VectorReplayF64.java
```

Nothing was committed or pushed in either repository. `nablatensor-web` has no
outstanding changes from this section (the pom.xml edit used to point at the local
SNAPSHOT for testing was reverted; `bench/lib-comparison/cp.txt` and `out/` are
regenerable build artifacts, not committed). The diff is correct (see verification
above) and left in place deliberately, not reverted — it is a real, working,
bit-exact-verified implementation of liveness-based row reuse that happens to be a
performance regression as written, which is itself the finding; reverting it would
throw away the one piece of runnable evidence for exactly how the indirection cost
arises. Whether to keep exploring this direction, adapt it (e.g. along the two lines
above), or drop it is a call for whoever owns that engine, not something to decide
here.

## Reproduction commands (Part 2)

```bash
# --- perf, both sides, steady-state window ---
cd bench/lib-comparison
sudo perf stat -D 1500 -e task-clock,cycles,instructions,cache-references,cache-misses -- \
  java -Xmx16g --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
  -Dengine=simd -Dproduct=asian -Dsteps=252 -Dpaths=300000 -Dgreeks=true -Dthreads=8 \
  -Dnablatensor.crn=on -Dladders=40 CrnLadder

cd bench/lib-comparison-nonjava
V=/tmp/…/scratchpad/aadc_probe_venv/bin/python   # the venv from Part 1
sudo perf stat -D 1000 -e task-clock,cycles,instructions,cache-references,cache-misses -- \
  env LADDERS=40 $V matlogica_bench.py asian 200000 greeks cached

# --- vSlotCount probe (reflection into the package-private field) ---
# VSlotProbe.java: builds an Asian MonteCarlo on engine=simd, runs it once, then
# reflects MonteCarlo.pricer.pricer.executable.vSlotCount (BatchedReplay's field)

# --- correctness diff: unmodified vs modified, both products, both passes, 1 & 8 threads ---
cd /home/petr/Projects/highprio/nablatensor && git stash   # unmodified
mvn -q -o -pl nablatensor-engine-simd -am install -DskipTests
# ... run CrnLadder for each (product, greeks, threads) combination, record price/delta ...
git stash pop                                              # modified
mvn -q -o -pl nablatensor-engine-simd -am install -DskipTests
# ... re-run the same combinations, diff against the recorded values ...

# --- bytecode tripwire, must still pass ---
cd /home/petr/Projects/highprio/nablatensor
mvn -q -o -pl nablatensor-examples test -Dtest=VectorSweepSplitTest

# --- JFR on the modified engine ---
java -Xmx16g --add-modules jdk.incubator.vector -cp "out:$(cat cp.txt)" \
  -XX:+FlightRecorder -XX:StartFlightRecording=filename=simd_modified.jfr,settings=profile \
  -Dengine=simd -Dproduct=asian -Dsteps=252 -Dpaths=300000 -Dgreeks=true -Dthreads=8 \
  -Dnablatensor.crn=on -Dladders=20 CrnLadder
jfr print --events jdk.ExecutionSample simd_modified.jfr | grep -c "Arrays.fill"
```
