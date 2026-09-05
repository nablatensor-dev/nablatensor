# `container/` — run the demos in a container

A self-contained image that runs the `demo/*.sh` narrated jshell sessions on
**Ubuntu 24.04** with **Azul Zulu JDK 26**, on any of the three engines that
work on this box: `cpu-jit`, `vulkan` (Mesa RADV), `rocm` (HIP).

```
container/
  Containerfile     the image
  build.sh          podman build wrapper
  run-demo.sh       podman run wrapper (GPU passthrough)
  _env.sh           shared: image name + the isolated podman store
  .containerignore  keeps the build context tiny
```

## Build

```bash
container/build.sh
```

What it does:

1. Compiles the project on the host if `*/target/classes` is missing
   (`mvn -o compile`) — the demos run jshell against those class dirs, so the
   image ships the compiled tree, not a JDK-less JRE and not Maven.
2. Stages the **minimal ROCm userspace** (~185 MB) out of the host's ~22 GB
   `/opt/rocm`: just `libamdhip64`, `libhiprtc`, `libhiprtc-builtins`,
   `libamd_comgr`, `libhsa-runtime64`, `librocprofiler-register`, the
   `amdgcn/bitcode` device libs and `rocminfo`.
3. `podman build` with two extra build contexts — `jdk=` (the Zulu 26 dir) and
   `rocm=` (the stage from step 2) — and the Vulkan userspace from `apt`.

### Why a separate podman store

This box's default rootless-podman store uses the kernel-native `overlay`
driver with `overlay.ignore_chown_errors`, under which `apt` cannot install
inside a build (it can't write the `_apt`-owned cache dirs). `build.sh` and
`run-demo.sh` therefore point podman at its own store under
`~/.local/share/nablatensor-podman` that uses **fuse-overlayfs** (already
installed), which represents multiple UIDs correctly. Your normal `podman`
store and config are untouched. Override the location with `$NT_PODMAN_ROOT`.

To see the image or clean up:

```bash
podman --root ~/.local/share/nablatensor-podman --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs images
rm -rf ~/.local/share/nablatensor-podman        # remove everything this created
```

## Run

```bash
container/run-demo.sh <engine> [command ...]
#   <engine> = cpu-jit | vulkan | rocm | -        (- lets the probe pick)
#   command  = default: interactive bash
```

Examples:

```bash
container/run-demo.sh vulkan  nt-gpucheck                          # show the GPUs
container/run-demo.sh cpu-jit ./demo/one-tape-every-backend.sh --fast
container/run-demo.sh vulkan  ./demo/isda-simm-full.sh --fast
container/run-demo.sh rocm    ./demo/one-tape-every-backend.sh --fast
```

GPU passthrough (rootless podman):

| engine | devices | extra |
|---|---|---|
| `cpu-jit` | none | — |
| `vulkan` | `/dev/dri/renderD128` | — |
| `rocm` | `/dev/dri/renderD128`, `/dev/kfd` | `--security-opt seccomp=unconfined` |

plus `--group-add keep-groups` so the container keeps the host's `render` /
`video` groups that own those nodes.

## Verifying the GPUs are real

`nt-gpucheck` inside the container prints the `vulkaninfo` / `rocminfo` device
lines and the library-footprint report. Beyond that, the demos themselves are
the proof: `one-tape-every-backend.sh` cross-checks every backend's price and
delta against the scalar CPU oracle and prints honest warm-then-measure
throughput, so a `vulkan` / `rocm` row that agrees with the oracle to ~5 d.p.
at millions of paths and posts a throughput well above `cpu-jit` is the GPU
actually executing.

## Library footprint (Vulkan vs ROCm)

Printed at build time and stored at `/opt/SIZES.txt` in the image
(`container/run-demo.sh - cat /opt/SIZES.txt`). Measured as the apparent-size
delta each layer adds on top of `ubuntu:24.04`. On this machine
(Radeon 780M, ROCm 7.2.4, Mesa 25.2.8):

| | added on top of ubuntu:24.04 | what dominates |
|---|---|---|
| **Vulkan userspace** | **~282 MiB** | `libllvm20` 137 MiB, `mesa-vulkan-drivers` 94 MiB (all ICDs), `libicu74` 35 MiB, `libshaderc1` 8 MiB |
| **ROCm userspace** | **~184 MiB** | `libamd_comgr.so` 153 MiB (HIPRTC's statically-linked LLVM), `libamdhip64.so` 26 MiB, `libhsa-runtime64.so` 4 MiB, `amdgcn` bitcode 1 MiB |

ROCm is staged down from the host's ~22 GB `/opt/rocm` to the ~185 MiB the HIP
runtime + runtime kernel compiler actually need; `apt` then adds only
`libnuma1`. The full image is ~1.0 GB (ubuntu + JDK 26 ≈ 480 MiB, then the two
GPU stacks).
