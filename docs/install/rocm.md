# Installing ROCm (for AMD GPUs)

`nablatensor-rocm` and `nablatensor-backend-rocm` call the HIP runtime
(`libamdhip64`) and its runtime kernel compiler (`libhiprtc`, which itself
depends on `libamd_comgr` — AMD's Code Object Manager, statically linking
LLVM) purely by `dlopen`, again through Java's Foreign Function & Memory API.
There is no vendor SDK compiled against, no `hipcc`/CMake step in the build —
only a working ROCm *userspace* and a kernel that exposes the right device
nodes need to exist on the machine that runs it. ROCm is AMD's platform only —
there is no ROCm path for Intel or NVIDIA hardware (the closest NVIDIA
equivalent, driven by `nablatensor-cuda`/`nablatensor-backend-cuda`, is CUDA,
which is a separate installation entirely and not covered here). The
[`container/`](../../container/README.md) directory shows one concrete,
already-working example (a Radeon 780M APU, ROCm 7.2.4) — this document
generalises it to any AMD GPU.

## Supported hardware

ROCm targets a specific, AMD-published list of GPUs per release — see AMD's
current [Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).
Broadly:

- **Officially supported**: Instinct-series data-center accelerators (MI50,
  MI100, MI2xx, MI3xx) and most Radeon Pro / recent Radeon RX consumer cards
  (RDNA2/RDNA3, e.g. RX 6800/6900, RX 7900).
- **Unsupported by AMD but frequently working anyway**: other RDNA2/RDNA3 GPUs,
  including many laptop APUs (e.g. the Radeon 780M this repository developed
  against, `gfx1103`), by forcing the runtime to treat the GPU as a supported
  architecture it is instruction-set-compatible with:

  ```bash
  # Query the real gfx target first (rocminfo prints "Name: gfxNNNN" per agent)
  export HSA_OVERRIDE_GFX_VERSION=11.0.0   # e.g. treat gfx110x as gfx1100 (supported)
  ```

  This is unsupported by AMD and can crash or silently miscompute on a genuine
  ISA mismatch — always validate against a CPU oracle (nablatensor's own
  `cpu`/`cpu-jit` backends are exactly such an oracle) after doing this, the
  same way this project's `one-tape-every-backend` demo cross-checks every
  backend's result before trusting its throughput number.
- **Not supported at all**: pre-GCN and most GCN1-3 GPUs, and anything older
  than what the installed ROCm release's LLVM backend still emits code for.

Find your GPU's `gfx` target and whether the installed ROCm build recognizes
it with `rocminfo | grep gfx` (see [Verifying the install](#verifying-the-install)).

## Host kernel requirements

Regardless of distro or install method, the **host** kernel needs:

- The `amdgpu` kernel module loaded (in-tree in any mainline kernel new enough
  for the GPU in question; very new GPUs may need a newer kernel than the
  distro ships, or the out-of-tree `amdgpu-dkms` package from AMD's repo).
- `/dev/dri/renderD1XX` exposed (shared with Vulkan/OpenCL/graphics — see
  [`vulkan.md`](vulkan.md)) **and** `/dev/kfd` exposed (ROCm-specific — this is
  the Kernel Fusion Driver node HSA/HIP use to manage queues and memory).
- The invoking user in the `render` group (owns `/dev/kfd` and the render
  nodes) and, on older distros, the `video` group too.

```bash
sudo usermod -aG render,video "$USER"   # re-login (or reboot) for it to take effect
ls -l /dev/kfd /dev/dri/renderD128       # confirm both exist and are group-accessible
```

## Installing the full stack

ROCm is **not** in the default Debian/Ubuntu/Fedora repositories — always
AMD's own repository, or the equivalent for Instinct-integrated distros
(SLES, RHEL). AMD publishes a single installer script/package that works the
same way across the supported distros:

```bash
# 1. Fetch the distro-specific installer package from repo.radeon.com, e.g. Ubuntu 24.04 / ROCm 7.2:
wget https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4-<rev>_all.deb
sudo apt-get install ./amdgpu-install_*.deb

# 2. Let it set up the repos, then install just the HIP/compute usecase
#    (--no-dkms skips the kernel module if the host kernel already has amdgpu built-in
#    or you manage it separately; drop it if you need amdgpu-dkms installed too)
sudo amdgpu-install --usecase=hiplibsdk --no-dkms
```

Equivalent RPM-based flow (Fedora/RHEL/SLES) fetches the matching `.rpm`
installer from the same `repo.radeon.com/amdgpu-install/<version>/...` tree and
runs `amdgpu-install` identically — the usecase names and script are shared
across distros; only the package format differs.

### Adding the repo manually (no installer script)

Equivalent to what the installer script does under the hood, useful in a
container build or anywhere you want the smallest possible `apt` transaction:

```bash
apt-get install -y wget gnupg
mkdir -p /etc/apt/keyrings
wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
  | gpg --dearmor > /etc/apt/keyrings/rocm.gpg
echo 'deb [signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.4 noble main' \
  > /etc/apt/sources.list.d/rocm.list
echo 'deb [signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/7.2.4/ubuntu noble main' \
  > /etc/apt/sources.list.d/amdgpu.list
apt-get update

# full stack (~5 GB, includes profilers, math libraries, rocminfo, etc.)
apt-get install -y rocm-hip-runtime rocminfo

# what nablatensor actually needs — HIP runtime + HIPRTC/comgr + HSA + rocminfo
apt-get install -y hip-runtime-amd hsa-rocr comgr rocm-llvm rocminfo
```

Replace `noble`/`7.2.4` with the codename and ROCm version matching your
distro release and the driver support you need — check
[repo.radeon.com](https://repo.radeon.com/rocm/apt/) for available versions.

`comgr` pulls in `rocm-llvm` (ROCm's own LLVM build, ~150 MB) because HIPRTC
compiles kernel source to device code through it at runtime — this is the
single largest piece of the whole stack and cannot be trimmed away for a
JIT-based consumer like nablatensor.

## Installing only what a HIP-runtime consumer needs (no compiler, no dev tools)

A full ROCm install is on the order of 20+ GB — mostly compilers (`hipcc`),
math libraries (rocBLAS, MIOpen, ...), profilers and debug tooling that a pure
runtime consumer like nablatensor never touches. If you already have a full
ROCm install on one machine (e.g. a build host) and want to ship only the
runtime pieces elsewhere (e.g. into a container image), the minimal set is:

```
lib/libamdhip64.so*              # HIP runtime
lib/libhiprtc.so*                # runtime kernel compiler front-end
lib/libhiprtc-builtins.so*       # HIPRTC's builtins
lib/libamd_comgr.so*             # Code Object Manager (statically links LLVM)
lib/libhsa-runtime64.so*         # HSA runtime underneath HIP
lib/librocprofiler-register.so*  # a small shim libamdhip64 dlopens unconditionally
amdgcn/bitcode/*                 # device-side bitcode libraries HIPRTC links against
bin/rocminfo                     # optional, but the standard way to verify agents
```

plus, from the distro's own repos rather than AMD's, the small set of
libraries these link against: `libnuma1`, `libdrm2`, `libdrm-amdgpu1`,
`libelf1` (Debian/Ubuntu package names; equivalent `libnuma`, `libdrm`,
`elfutils-libelf` etc. on RPM distros).

`container/build.sh` in this repository does exactly this: it stages those
~185 MB out of a full host `/opt/rocm` into a temporary directory and hands it
to `podman build` as a build context, rather than copying all ~22 GB in.

## Verifying the install

```bash
rocminfo | grep -E "^\s*Name:|Marketing Name:|Device Type:|gfx"
```

Each GPU shows up as an "Agent" with `Device Type: GPU` and a `Name: gfxNNNN`
line — that `gfxNNNN` is the ISA target to compare against ROCm's supported
list, and what you'd override with `HSA_OVERRIDE_GFX_VERSION` if targeting an
unsupported-but-compatible chip. A missing GPU agent (only a CPU agent listed)
usually means `/dev/kfd` isn't accessible, `amdgpu` isn't loaded, or the ROCm
build doesn't recognize this GPU's PCI ID at all.

## Environment variables

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib   # only needed if not on the system ld.so path
```

If installed system-wide via the distro packages above, `ldconfig` picks up
`/opt/rocm/lib` automatically (the installer drops a `/etc/ld.so.conf.d/rocm.conf`
or equivalent) and none of this is strictly required; it matters most when
running from a hand-staged directory as described above.

## Running inside a container

Beyond installing the userspace pieces (above) into the image, container
device passthrough for ROCm needs, regardless of runtime:

- `--device /dev/dri/renderD1XX --device /dev/kfd` (both — HIP needs `/dev/kfd`
  for queue/memory management even though the actual command submission also
  touches the render node).
- Group membership preserved so the containerized process can open those
  nodes — e.g. Podman's `--group-add keep-groups` (this requires the `crun`
  runtime, not `runc`, to actually propagate supplementary groups), or an
  explicit `--group-add <host-render-gid>` under Docker.
- `--security-opt seccomp=unconfined` in many current setups — HSA's queue
  doorbell mechanism uses syscalls (e.g. certain `ioctl`/futex patterns) that
  default container seccomp profiles can block; a properly scoped custom
  seccomp profile is the tighter alternative to disabling it outright, but
  unconfined is the common pragmatic default.

`container/Containerfile` and `container/run-demo.sh` in this repository show
a complete worked example of all of the above alongside a Vulkan install in
the same image.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `rocminfo` shows only a CPU agent, no GPU | `/dev/kfd` missing/inaccessible, `amdgpu` not loaded, or this GPU's PCI ID isn't in this ROCm build's support table at all |
| `rocminfo` shows the GPU but nablatensor's `rocm` backend reports itself unavailable | `libamdhip64.so`/`libhiprtc.so` not on the library path, or a required linked library (`libnuma1`, `libdrm2`, ...) missing |
| HIPRTC compile fails or crashes only on this GPU | Unsupported `gfx` target — try `HSA_OVERRIDE_GFX_VERSION` set to the nearest supported architecture, and validate results against a CPU backend afterward |
| Works as root, fails as a normal user | User not in the `render`/`video` group; re-login after `usermod -aG render,video $USER` |
| Works on the host, fails only in a container | `/dev/kfd` not passed through, group not preserved into the container, or seccomp blocking HSA — see above |
