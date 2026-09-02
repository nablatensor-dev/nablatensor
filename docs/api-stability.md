# API stability

NablaTensor is pre-1.0; signatures may still change. This page says which parts
are meant to be built on and which are not.

## What is API

Application code should depend only on **unannotated public types in these
packages**:

| Package | Surface |
|---|---|
| `com.nablatensor.engine` | `Nabla`, `SDouble`, `AadRecorder`, `AadTape`, `AadResult`, `AadOptions`, `JitOptimizations` |
| `com.nablatensor.engine` (SPI) | `AadEngine`, `AadExecutable` — implement to add a backend |
| `com.nablatensor.quant` | `MonteCarlo`, `Product`, `TimeGrid`, `EquityMarket` (and the other `*Market` records), `Products`, `ExoticProducts`, `Hooks`, `MultiOutput`, `MultiMetric`, `Calibrator`, `BlackScholes`, the `*Model` step blocks |
| `com.nablatensor.scenario` | `Shock`, `Scenario`, `ScenarioSet`, `Ladder`, `ScenarioRunner` |
| `com.nablatensor.risk` | the aggregation types |
| `com.nablatensor.tensor` | `NablaTensors`, `Tensor`, `Shape`, `Device`, `DType`, `PrngKey`, and `com.nablatensor.tensor.spi.*` |

## What is not

- **`@com.nablatensor.annotation.Internal`** — a public type, member or package
  that is public only so another `nablatensor-*` module can reach it. It may
  change or vanish in any release, with no deprecation cycle. Examples:
  `AbstractAadExecutable`, `CudaAadCodegen`, `AadCheckpointPlan`, the per-backend
  `*Replay` / `*Kernel` / `*Codegen` classes.
- **`nablatensor-backend-{cuda,rocm,vulkan}`** — the low-level device runtimes the
  GPU replay engines dispatch through. Entirely internal.
- **Most of `nablatensor-tensor` beyond the types listed above** — the module
  exists for the replay backends and the MNIST-scale example. `BackendRegistry`,
  `DenseLinalg`, `Jit`, `Linalg`, `Expr`, `ExprBuffer`, `FusingBackend`,
  `TreeDef`, `TreeUtil`, `ConvSpec` are internal.

## Why no `module-info.java`

A JPMS module descriptor would enforce the boundary at compile time, but:

1. `nablatensor-cuda` currently shares the `com.nablatensor.engine` package with
   `nablatensor-core` (a split package JPMS forbids). Relocating that module into
   `com.nablatensor.engine.cuda` is a prerequisite and has not been done.
2. `Nabla.model(marketRecord, …)` reflects over the **caller's** record type
   (`MarketShape`), which under JPMS requires the caller to `opens` its own
   package to `nablatensor.core` — friction pushed onto every user.

So the boundary is convention (`@Internal` + this page) for now. The `.internal`
subpackage split and a later `module-info` remain open follow-ups.
