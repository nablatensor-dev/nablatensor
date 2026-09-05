# `container/` — run the demos in a container

A self-contained image that runs the `demo/*.sh` narrated jshell sessions on
**Ubuntu 24.04** with **Azul Zulu JDK 26**. As shipped, it runs the `cpu-jit`
engine only — no GPU driver, vendor library or extra image size baked in.

For the general Vulkan/ROCm installation story on any GPU vendor — what to
install for AMD/Intel/NVIDIA Vulkan, which ROCm packages a plain HIP-runtime
consumer actually needs, device passthrough into a container, and
troubleshooting — see [`docs/install/vulkan.md`](../docs/install/vulkan.md)
and [`docs/install/rocm.md`](../docs/install/rocm.md). To turn this image into
a GPU-enabled one, see
[`docs/install/gpu-container.md`](../docs/install/gpu-container.md), which
walks through extending the `Containerfile` below with a GPU stage, worked
through with this project's own hardware (AMD Radeon 780M, Vulkan + ROCm) as
the example, plus how to run the result.

```
container/
  Containerfile     the image (CPU-only)
  build.sh          podman build wrapper
  run-demo.sh       podman run wrapper (device passthrough, incl. for a
                     GPU-enabled image you build per docs/install/gpu-container.md)
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
2. `podman build` with one extra build context — `jdk=` (the Zulu 26 dir).

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

```bash
container/run-demo.sh cpu-jit ./demo/one-tape-every-backend.sh --fast
container/run-demo.sh cpu-jit ./demo/isda-simm-full.sh --fast
```

`vulkan` and `rocm` need an image that actually has that GPU stack installed —
see [`docs/install/gpu-container.md`](../docs/install/gpu-container.md) for how
to build one and the exact `podman run` invocations for both engines.
