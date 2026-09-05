# Building a GPU-enabled Containerfile

`container/Containerfile` (built by `container/build.sh`) ships **CPU-only**:
Ubuntu 24.04 + the JDK, nothing else. That is deliberate — which GPU packages
to install depends on the vendor and model of the GPU the *image* will
eventually run on, which is a fact about the deployment machine, not about
this project, so it doesn't belong hard-coded into the one Containerfile
everybody builds. [`vulkan.md`](vulkan.md) and [`rocm.md`](rocm.md) in this
same directory cover that installation story in general, for every vendor.

This document is the other half: how to turn the base Containerfile into a
GPU-enabled one — add a stage, rebuild, run with device passthrough — worked
through concretely with **this project's own development hardware** (an AMD
Radeon 780M integrated GPU: Vulkan via Mesa RADV, ROCm via HIP) as the
example. Swap the package names in step 2 for whatever `vulkan.md`/`rocm.md`
says for your vendor and the rest of this document is unchanged.

## The shape of the change

Everything below is added into `container/Containerfile`, between the JDK
stage and the "copy the project in" stage that are already there. Nothing
about the rest of the Containerfile, `build.sh`, or `run-demo.sh` needs to
change — `run-demo.sh` already knows how to pass `vulkan`/`rocm` device flags
to whatever image `$NT_IMAGE` names (see `container/run-demo.sh`); it only
needs an image that actually has the corresponding userspace installed.

## Step 1 — Vulkan (works unchanged for any vendor)

Vulkan's install is vendor-neutral at the package level — see
[`vulkan.md`](vulkan.md) for AMD/Intel/NVIDIA/software specifics — so this
stage rarely needs edits beyond swapping in an NVIDIA driver package if
targeting NVIDIA:

```dockerfile
# --- Vulkan userspace -------------------------------------------------------
# Mesa's RADV (AMD) / ANV (Intel) ICDs both live in mesa-vulkan-drivers; a
# host with an NVIDIA GPU needs the NVIDIA driver package instead — see
# docs/install/vulkan.md.
RUN apt-get update && apt-get install -y --no-install-recommends \
        mesa-vulkan-drivers libvulkan1 libshaderc1 vulkan-tools \
 && rm -rf /var/lib/apt/lists/*
```

`vulkan-tools` is optional — it is only the `vulkaninfo` diagnostic used to
verify the device is visible (see [Verifying](#verifying-the-gpus-are-real)
below).

## Step 2 — ROCm (AMD only)

Two ways to bring ROCm into the image; pick one.

### Option A — apt from AMD's repo (simplest, self-contained)

No host staging, no extra build context — just add AMD's repo and install the
runtime packages a HIP-runtime consumer needs (see [`rocm.md`](rocm.md) for
what each package is for and the RPM-based equivalent):

```dockerfile
# --- ROCm / HIP runtime ------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg \
 && mkdir -p /etc/apt/keyrings \
 && wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
      | gpg --dearmor > /etc/apt/keyrings/rocm.gpg \
 && echo 'deb [signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.4 noble main' \
      > /etc/apt/sources.list.d/rocm.list \
 && echo 'deb [signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/7.2.4/ubuntu noble main' \
      > /etc/apt/sources.list.d/amdgpu.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      hip-runtime-amd hsa-rocr comgr rocm-llvm rocminfo \
 && rm -rf /var/lib/apt/lists/*
ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib
```

This pulls in `rocm-llvm` (~150 MB, HIPRTC's kernel compiler backend) fresh
from AMD's repo — simple, but the biggest part of the image is downloaded
every build.

### Option B — stage a minimal runtime from an existing host ROCm install

If a machine with a full ROCm install (~20+ GB) is already available (e.g. the
build host), this avoids the ~150 MB `rocm-llvm` download by copying just the
~185 MB actually needed for a HIP-runtime consumer, the same way this
project's Containerfile did before this GPU stage was pulled out of it:

```dockerfile
# --- ROCm / HIP runtime (staged from an extra build context, see below) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnuma1 libdrm2 libdrm-amdgpu1 libelf1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=rocm . /opt/rocm/
ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib
RUN echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf && ldconfig
```

which needs a staging step before `podman build`, populating a temp directory
with just the runtime pieces (add this to `container/build.sh`, or run it by
hand once):

```bash
ROCM_DIR=/opt/rocm
stage=$(mktemp -d)
mkdir -p "$stage/lib" "$stage/bin"
for so in libamdhip64 libhiprtc libhiprtc-builtins libamd_comgr \
          libhsa-runtime64 librocprofiler-register; do
  cp -aP "$ROCM_DIR"/lib/${so}.so* "$stage/lib/"
done
cp -a "$ROCM_DIR/amdgcn" "$stage/"
cp -a "$ROCM_DIR/bin/rocminfo" "$stage/bin/"
```

then pass it as a build context (see [Building](#building) below).

### A note on unsupported-but-similar GPUs (e.g. this project's Radeon 780M)

The Radeon 780M's `gfx1103` is not in ROCm's officially supported list.
Ordinarily that means passing `HSA_OVERRIDE_GFX_VERSION` at runtime to make
the ROCm runtime treat the agent as a supported architecture (see
[`rocm.md`](rocm.md#supported-hardware)). nablatensor's own ROCm backend
(`HipRuntime`/`HipCompute` in `nablatensor-backend-rocm`) already does the
equivalent at the application level — it detects an unrecognized `gfxNNNN`
token and compiles the HIPRTC kernel with `--offload-arch=gfx1100` instead of
the literal reported target — so running nablatensor itself against this
image needs **no** `HSA_OVERRIDE_GFX_VERSION` set. That variable only matters
if something *else* in the image (e.g. `rocminfo`-adjacent tooling) needs the
override too.

## Building

```bash
# Option A (apt-only ROCm): no extra context beyond the existing jdk= one
podman build --build-context jdk="$JDK_DIR" -t nablatensor-demo:latest \
  -f container/Containerfile .

# Option B (staged ROCm): add the rocm= context from the staging step above
podman build --build-context jdk="$JDK_DIR" --build-context rocm="$stage" \
  -t nablatensor-demo:latest -f container/Containerfile .
```

Using this project's own isolated fuse-overlayfs podman store (see
`container/README.md` for why), that's the same invocation `container/build.sh`
already uses, e.g.:

```bash
podman --root ~/.local/share/nablatensor-podman \
       --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs \
       --runtime "$PWD/container/.bin/crun" \
  build --build-context jdk="$JDK_DIR" \
        --ignorefile container/.containerignore \
        -t nablatensor-demo:latest -f container/Containerfile .
```

## Running

Device passthrough is the same regardless of which ROCm option was used to
build the image. `container/run-demo.sh vulkan|rocm ...` already wraps this
(see `container/README.md`); spelled out in full against the image built
above, on this project's own hardware:

**Vulkan** (`/dev/dri/renderD128` — adjust the number if this isn't the
right render node; find it with `ls /dev/dri/`):

```bash
podman --root ~/.local/share/nablatensor-podman \
       --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs \
       --runtime "$PWD/container/.bin/crun" \
  run --rm -it --init --group-add keep-groups \
      -e HOME=/root -e TERM=xterm -w /opt/nablatensor \
      -e NABLATENSOR_DEMO_ENGINE=vulkan \
      --device /dev/dri/renderD128 \
      nablatensor-demo:latest ./demo/isda-simm-full.sh --fast
```

**ROCm** (`/dev/dri/renderD128` + `/dev/kfd`, plus an unconfined seccomp
profile — HSA's queue-doorbell mechanism needs syscalls the default profile
blocks):

```bash
podman --root ~/.local/share/nablatensor-podman \
       --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs \
       --runtime "$PWD/container/.bin/crun" \
  run --rm -it --init --group-add keep-groups \
      -e HOME=/root -e TERM=xterm -w /opt/nablatensor \
      -e NABLATENSOR_DEMO_ENGINE=rocm \
      --device /dev/dri/renderD128 --device /dev/kfd \
      --security-opt seccomp=unconfined \
      nablatensor-demo:latest ./demo/one-tape-every-backend.sh --fast
```

`--runtime .../crun` matters here specifically because `--group-add
keep-groups` (needed so the container process keeps the host's `render`/
`video` group membership that owns those device nodes) is a `crun`-only
feature — the default `runc` cannot propagate supplementary groups. See
`container/_env.sh`, which sets this up automatically for `container/build.sh`
/ `container/run-demo.sh` and fetches a static `crun` if none is installed.

## Verifying the GPUs are real

Add a small diagnostic script to the image if useful:

```dockerfile
RUN printf '%s\n' '#!/usr/bin/env bash' \
  'echo "=== Vulkan (vulkaninfo) ==="' \
  'vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName|driverInfo|apiVersion|deviceType" || echo "  no Vulkan device visible"' \
  'echo; echo "=== ROCm (rocminfo) ==="' \
  '/opt/rocm/bin/rocminfo 2>/dev/null | grep -E "^\s*Name:|Marketing Name:|Device Type:|gfx" | head -24 || echo "  no ROCm agent visible"' \
  > /usr/local/bin/nt-gpucheck && chmod +x /usr/local/bin/nt-gpucheck
```

then `container/run-demo.sh vulkan nt-gpucheck` (or the raw `podman run`
equivalent above with `nt-gpucheck` instead of a demo script) prints the
enumerated devices. Beyond that, the demos themselves are the real proof:
`demo/one-tape-every-backend.sh` cross-checks every backend's price and delta
against the scalar CPU oracle and prints honest warm-then-measure throughput,
so a `vulkan`/`rocm` row that agrees with the oracle to ~5 d.p. at millions of
paths and posts a throughput well above `cpu-jit` is the GPU actually
executing, not a silent fallback.
