# Taming `GpuKernels.TENSOR_SOURCE`: seven ways to make the C block Java-like

## The situation

`nablatensor-tensor/src/main/java/com/nablatensor/tensor/spi/GpuKernels.java` is the single
place where the CUDA and ROCm/HIP tensor backends get their device code. It is shared on
purpose — NVRTC and HIPRTC both accept the same CUDA-C subset, so an op added here lands on
both GPU flavours at once.

The cost of that decision is one enormous artifact:

| Element | Today |
| --- | --- |
| `TENSOR_SOURCE` | one text block, ~360 lines, 20 `extern "C" __global__` kernels concatenated |
| `TENSOR_KERNEL_NAMES` | a hand-maintained `String[]` that must stay in sync with the block |
| `fusedSource(...)` | a `StringBuilder` that hand-splices `extern "C" __global__ void ...` |
| Consumers | `CudaBackend#loadModule` and `RocmBackend#loadModule`, both doing `compile(TENSOR_SOURCE, arch)` then looping the name array |

Three concrete pains:

1. **No structure.** Finding `reduce_axis_argmax` means scrolling a 360-line string literal;
   there is no navigation, no outline, no per-kernel Javadoc.
2. **Two sources of truth.** The kernel names live both in the C text and in
   `TENSOR_KERNEL_NAMES`. A typo or a forgotten entry is only caught at module-load time on a
   machine with a GPU.
3. **Zero tooling.** No syntax highlighting, no clang-format, no per-kernel test, no way to say
   "this kernel needs block dim 16×16" other than remembering it at the call site.

What follows is seven independent approaches. They are not a sequence — pick one, or combine
1+4, or 2+6. Each section states the shape, a sketch, what it buys, what it costs, and a
rough risk rating.

---

## Approach 1 — One `.cu` resource file per kernel, loaded from the classpath

**Shape.** Delete the text block. Each kernel becomes a real file under
`nablatensor-tensor/src/main/resources/com/nablatensor/tensor/kernels/*.cu`. Java keeps only
the manifest and the concatenation.

```
nablatensor-tensor/src/main/resources/com/nablatensor/tensor/kernels/
  ew_binary.cu
  ew_scalar.cu
  matmul_tiled.cu
  reduce_axis_argmax.cu
  ...
```

```java
public final class GpuKernels {

  public static final String[] TENSOR_KERNEL_NAMES = { "ew_binary", /* ... */ };

  public static final String TENSOR_SOURCE = Arrays.stream(TENSOR_KERNEL_NAMES)
      .map(GpuKernels::readKernel)
      .collect(Collectors.joining("\n"));

  private static String readKernel(String name) {
    String path = "kernels/" + name + ".cu";
    try (InputStream in = GpuKernels.class.getResourceAsStream(path)) {
      if (in == null) {
        throw new IllegalStateException("missing kernel resource: " + path);
      }
      return new String(in.readAllBytes(), StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }
}
```

**Buys.** Real `.cu` files get IDE syntax highlighting, clang-format, `nvcc --dryrun`
compile-checks in CI without a GPU, and sane `git blame`/diffs. `GpuKernels.java` shrinks to
about 60 lines of actual Java. The array becomes the manifest, and a missing file fails loudly
at class-init instead of at `cuModuleGetFunction`.

**Costs.** Device code leaves the `.java` file, so a reader of `GpuKernels` no longer sees the
kernels inline. Resource loading must survive shading/jlink/native-image packaging — a
`module-info`/`maven-shade` misconfiguration turns a compile-time certainty into a runtime
`NullPointerException`. Ordering is now the array's job, so `#include`-style dependencies
between kernels have to be modelled explicitly.

**Risk.** Low. Purely mechanical; the emitted string can be asserted byte-identical to today's
in a one-off test.

---

## Approach 2 — One nested `record` per kernel, each carrying its own text block

**Shape.** Keep everything in Java, but stop having one string. Model a kernel as a value.

```java
public record GpuKernel(String name, String source, int blockDimX, int blockDimY) {

  public GpuKernel {
    Objects.requireNonNull(name);
    if (!source.contains(name)) {
      throw new IllegalArgumentException("source does not define " + name);
    }
  }
}
```

```java
public final class GpuKernels {

  /** Elementwise binary op dispatch; `op` indexes {@link Op} in declaration order. */
  static final GpuKernel EW_BINARY = new GpuKernel("ew_binary", """
      extern "C" __global__ void ew_binary(float* out, const float* a, const float* b,
          int n, int op) {
        ...
      }
      """, 256, 1);

  /** 16x16 shared-memory tiled GEMM; grid is chunked via `tile_offset`. */
  static final GpuKernel MATMUL_TILED = new GpuKernel("matmul_tiled", """
      ...
      """, 16, 16);

  public static final List<GpuKernel> TENSOR_KERNELS = List.of(EW_BINARY, MATMUL_TILED, /* ... */);

  public static final String TENSOR_SOURCE = TENSOR_KERNELS.stream()
      .map(GpuKernel::source)
      .collect(Collectors.joining("\n"));
}
```

**Buys.** This is the most "Java-like" answer with the least ceremony. Each kernel gets a name
in the IDE outline, its own Javadoc, and — crucially — its launch geometry travels with it
instead of being a magic `16` at the call site. The name/source consistency check moves into
the record's compact constructor, so it runs at class-init on every JVM, GPU or not. Backends
can iterate `TENSOR_KERNELS` and read `blockDimX()` rather than hardcoding.

**Costs.** The C is still inside `.java`, so still no highlighting or clang-format. The file
stays long (~420 lines), just navigable. Text blocks nested in a constructor argument indent
awkwardly, and `\"` escaping inside `extern "C"` remains.

**Risk.** Low. No build changes at all.

---

## Approach 3 — An annotation-driven kernel registry (`@GpuKernel`), assembled by reflection

**Shape.** The user's "put some parts into annotations" idea taken literally: metadata goes to
annotations, the body stays a constant.

```java
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface GpuKernel {
  String name();
  int blockDimX() default 256;
  int blockDimY() default 1;
  int order() default 0;
  String[] requires() default {};  // other kernel names this one calls
}
```

```java
final class TensorKernels {

  @GpuKernel(name = "ew_binary", order = 10)
  static final String EW_BINARY = """
      extern "C" __global__ void ew_binary(...) { ... }
      """;

  @GpuKernel(name = "matmul_tiled", order = 50, blockDimX = 16, blockDimY = 16)
  static final String MATMUL_TILED = """
      extern "C" __global__ void matmul_tiled(...) { ... }
      """;
}
```

```java
public static String tensorSource() {
  return Arrays.stream(TensorKernels.class.getDeclaredFields())
      .filter(f -> f.isAnnotationPresent(GpuKernel.class))
      .sorted(Comparator.comparingInt(f -> f.getAnnotation(GpuKernel.class).order()))
      .map(GpuKernels::readConstant)
      .collect(Collectors.joining("\n"));
}
```

**Buys.** Declarative and idiomatic-looking: adding a kernel is one annotated field, and
`TENSOR_KERNEL_NAMES` disappears entirely (derive it from the annotations). Metadata is
extensible without touching call sites — add `sharedMemBytes()`, `minArch()`,
`@GpuKernel(cudaOnly = true)` later. The `requires()` field lets the assembler topologically
sort device-function dependencies instead of relying on field order.

**Costs.** Reflection over declared fields is fragile under shading, obfuscation, JPMS
(`opens` needed), and GraalVM native-image (needs reflect-config). Field ordering from
`getDeclaredFields()` is *not specified* by the JVM spec, which is exactly why `order()` has
to exist — a hand-maintained integer, i.e. the sync problem in a new costume. It is also a lot
of machinery for a list of 20 things.

**Risk.** Medium. Consider Approach 4 instead if you want annotations without reflection.

---

## Approach 4 — `@GpuKernel` plus an annotation processor that generates the source at compile time

**Shape.** Same annotations as Approach 3, but an `javax.annotation.processing.Processor`
reads them at `mvn compile` and emits a `GeneratedTensorKernels` class (or a `.cu` resource).
Nothing reflective survives to runtime.

```java
@SupportedAnnotationTypes("com.nablatensor.tensor.spi.GpuKernel")
public final class GpuKernelProcessor extends AbstractProcessor {

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment env) {
    // 1. collect @GpuKernel fields, read their constant values via
    //    VariableElement#getConstantValue()
    // 2. validate: name matches the `__global__ void <name>(` in the body,
    //    no duplicate names, every `requires()` target exists
    // 3. emit GeneratedTensorKernels.TENSOR_SOURCE and .TENSOR_KERNEL_NAMES
    return true;
  }
}
```

**Buys.** Every consistency rule that is a runtime surprise today becomes a **compile error**:
missing kernel, name mismatch, duplicate entry, unsorted dependency. Zero runtime cost and
zero reflection — the generated class is a plain constant, so shading/native-image are
non-issues. The processor is also the natural home for a "reject syntax CUDA accepts but HIP
does not" lint, which is the actual portability risk this class exists to manage.

**Costs.** The heaviest option by far: a new module, a `META-INF/services` registration, and
processor debugging (which is its own genre of pain). Contributors now need to understand a
codegen step to add a kernel. `VariableElement#getConstantValue()` only works for `static
final String` constant expressions, so text blocks are fine but concatenation of non-constants
is not.

**Risk.** High effort, low runtime risk. Only worth it if the kernel count grows well past 20
or per-backend variants proliferate.

---

## Approach 5 — A typed CUDA-C builder DSL (extend what `fusedSource` already does)

**Shape.** `fusedSource`/`emitFused`/`binaryExpr`/`unaryExpr` already prove the codebase can
*generate* correct device code from Java. Promote that from a private helper to the way all
kernels are written.

```java
static final GpuKernel EW_UNARY = CudaC.kernel("ew_unary")
    .param(CudaC.ptr("out"))
    .param(CudaC.constPtr("a"))
    .param(CudaC.i32("n"))
    .param(CudaC.i32("op"))
    .body(b -> {
      var i = b.declare("i", "blockIdx.x * blockDim.x + threadIdx.x");
      b.guard(i + " >= n");
      var x = b.declareFloat("x", "a[" + i + "]");
      b.switchOn("op", sw -> {
        for (Op op : Op.unaryOps()) {
          sw.caseOf(op.ordinal(), "out[" + i + "] = " + unaryExpr(op, x) + ";");
        }
      });
    })
    .build();
```

**Buys.** The most Java-like of all seven: no string literals, refactorable, and the *same*
`unaryExpr`/`binaryExpr` helpers feed both `ew_unary` and `fusedSource`, so the op semantics
(the NaN-propagating `fmaxf` dance, say) are defined exactly once instead of twice as they are
today. A second emitter backend (GLSL for Vulkan, or SPIR-V) becomes possible later from the
same tree.

**Costs.** You are writing a C compiler front-end in Java for 20 kernels. Kernels with
`__shared__ float tile_a[16][16]` and nested tiling loops (`matmul_tiled`,
`batched_matmul_tiled`, the three `conv2d_*`) will fight the DSL and end up as raw escape
hatches anyway, giving you the worst of both. Debugging shifts from "read the C" to "read the
Java that prints the C".

**Risk.** High. Recommended *only* for the elementwise/reduction family, with tiled and conv
kernels left as literal source (Approach 1 or 2).

---

## Approach 6 — Split by family into package-private classes; `GpuKernels` becomes a façade

**Shape.** Don't change the representation at all; change the file layout. The 20 kernels fall
into five obvious families.

```
com/nablatensor/tensor/spi/kernels/
  ElementwiseKernels.java   // ew_binary, ew_scalar, ew_unary, relu_backward
  MatmulKernels.java        // transpose2d, matmul_tiled, batched_matmul_tiled
  ReductionKernels.java     // reduce_sum, reduce_max, sum_axis0, reduce_axis_*, broadcast_to
  ConvKernels.java          // conv2d_fwd, conv2d_dx, conv2d_dw
  RandomKernels.java        // random_uniform, random_normal
```

```java
public final class GpuKernels {

  public static final String TENSOR_SOURCE = String.join("\n",
      ElementwiseKernels.SOURCE,
      MatmulKernels.SOURCE,
      ReductionKernels.SOURCE,
      ConvKernels.SOURCE,
      RandomKernels.SOURCE);

  public static final String[] TENSOR_KERNEL_NAMES = Stream.of(
          ElementwiseKernels.NAMES, MatmulKernels.NAMES, ReductionKernels.NAMES,
          ConvKernels.NAMES, RandomKernels.NAMES)
      .flatMap(Arrays::stream)
      .toArray(String[]::new);
}
```

**Buys.** Cheapest meaningful win in the list. No new concepts, no build changes, no runtime
behaviour change, and `TENSOR_KERNEL_NAMES` stops being one flat hand-maintained list —
each family owns its own short one next to its own source. Each family file is 40–90 lines,
which is readable. Also the natural seam for the "split a kernel back into the backend that
needs a variant" escape hatch the current class doc already describes: a family file can be
overridden wholesale.

**Costs.** Still text blocks, so still no highlighting or C tooling. The name↔source
duplication persists, just five times smaller each. Adds five files for what is arguably a
folding problem.

**Risk.** Very low. Good first step before any of 1–5.

---

## Approach 7 — Keep the text, add a parser-backed `KernelCatalog` that derives everything

**Shape.** Accept that the C block is the single source of truth, and delete the redundancy by
*reading* it instead of duplicating it.

```java
public record KernelSignature(String name, List<String> paramTypes, String source) { }

public final class KernelCatalog {

  private static final Pattern ENTRY = Pattern.compile(
      "extern\\s+\"C\"\\s+__global__\\s+void\\s+(\\w+)\\s*\\(([^)]*)\\)", Pattern.DOTALL);

  public static List<KernelSignature> parse(String translationUnit) { /* ... */ }
}
```

```java
public static final List<KernelSignature> TENSOR_KERNELS = KernelCatalog.parse(TENSOR_SOURCE);

public static final String[] TENSOR_KERNEL_NAMES = TENSOR_KERNELS.stream()
    .map(KernelSignature::name)
    .toArray(String[]::new);
```

**Buys.** `TENSOR_KERNEL_NAMES` stops existing as hand-written data, so it can never drift
from the source — the class's single worst failure mode is gone for the price of one regex.
The parsed parameter list unlocks a real payoff: backends can *validate launch arguments*
(arity, and `float*` vs `int`) in Java before `cuLaunchKernel`, turning a class of silent
memory corruption into an `IllegalArgumentException` with a kernel name in it. Composes with
every other approach here, including doing nothing else.

**Costs.** Regex-parsing C is only sound because this file's style is rigidly uniform; a
kernel written with a line break in the wrong place, a `template<>`, or a `__global__` inside a
comment silently produces a wrong catalog. Needs a golden test pinning the parsed names to the
expected 20, or it trades a loud failure for a quiet one.

**Risk.** Low–medium, entirely dependent on having that golden test.

---

## Comparison

| # | Approach | Effort | Fixes name drift | C tooling | Runtime risk |
| --- | --- | --- | --- | --- | --- |
| 1 | `.cu` classpath resources | Low | Partly (missing file fails) | Yes | Packaging |
| 2 | `record GpuKernel` per kernel | Low | Yes (constructor check) | No | None |
| 3 | `@GpuKernel` + reflection | Medium | Yes | No | Reflection/JPMS |
| 4 | `@GpuKernel` + annotation processor | High | Yes, at compile time | No | None |
| 5 | CUDA-C builder DSL | Very high | N/A (no names) | N/A | Generator bugs |
| 6 | Split by family | Very low | Reduces | No | None |
| 7 | Parse the source into a catalog | Low | Yes | No | Regex brittleness |

## Suggested combination — implemented

**6 → 2 → 7.** Split into family files first (no risk, immediate readability), then make each
family a `List<GpuKernel>` so launch geometry stops being a magic number at the call site,
then derive the name array by parsing so it can never drift. That reaches most of the benefit
without a codegen step or a packaging change.

Go to **1** instead of 2 if C tooling (clang-format, GPU-less `nvcc --dryrun` CI checks) is
what you actually want. Go to **4** only if the kernel count roughly doubles or CUDA/HIP
variants start diverging per kernel. Reserve **5** for the elementwise family, where it also
removes the real duplication between `ew_unary` and `fusedSource`.

### What landed

- `GpuKernel` — record of `(name, paramTypes, source, blockDimX, blockDimY)`. `GpuKernel.of`
  parses the single `extern "C" __global__` entry point out of the source, so the name and the
  signature are never typed twice.
- `ElementwiseKernels`, `MatmulKernels`, `RandomKernels`, `ReductionKernels`, `ConvKernels` —
  package-private, one `List<GpuKernel>` each. `DevicePrelude` holds the `__device__` helpers
  that more than one family calls, emitted ahead of everything.
- `GpuKernels` — now a façade: `TENSOR_KERNELS` (the flattened list), `TENSOR_SOURCE` (prelude
  plus every kernel's source), `TENSOR_KERNEL_NAMES` (derived from the parsed names, and
  checked for duplicates at class-init), plus `kernel(String)` for lookup. `fusedSource` and
  its emitter are unchanged.
- `CudaBackend` and `RocmBackend` take the 16×16 matmul launch geometry from
  `GpuKernels.kernel("matmul_tiled")` instead of hardcoding it; `grid16` became `gridTiles`.
- `GpuKernelsTest` pins the 20 parsed names, so a kernel that stops being recognised fails the
  build rather than disappearing from the module.

The emitted translation unit is line-for-line the same set of lines as before the split; only
the order changed (prelude first, `relu_backward` moved next to the other elementwise kernels).
