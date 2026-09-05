# Installing Vulkan (for any GPU)

`nablatensor-vulkan` and the tensor backend `nablatensor-backend-vulkan` talk to
whatever GPU is in the box through the standard Vulkan loader — they `dlopen`
`libvulkan.so.1` and `libshaderc.so.1` (via Java's Foreign Function & Memory
API) and never link against a vendor SDK. This means the *installation* step is
entirely a host/OS concern, identical for every project that uses Vulkan
compute, not something specific to nablatensor. This document covers that step
for AMD, Intel, NVIDIA and software-only (no GPU) setups. The
[`container/`](../../container/README.md) directory shows one concrete,
already-working example (Mesa RADV on an AMD APU) — this document generalises
it.

## How Vulkan is laid out on Linux

Three pieces, all replaceable independently:

1. **The loader** — `libvulkan.so.1` (package usually called `libvulkan1` or
   `vulkan-loader`). A thin, vendor-neutral dispatcher. Applications (including
   nablatensor) only ever call into the loader.
2. **One or more ICDs** (Installable Client Drivers) — the actual driver, e.g.
   `libvulkan_radeon.so` (Mesa RADV, AMD), `libvulkan_intel.so` (Mesa ANV,
   Intel), `libGLX_nvidia.so`/`libnvidia-vulkan-*` (NVIDIA proprietary), or
   `libvulkan_lvp.so` (Mesa **lavapipe**, a CPU software rasterizer used as a
   last resort or in CI with no GPU at all). Each ICD ships a small JSON
   manifest under `/usr/share/vulkan/icd.d/*.json` that just points the loader
   at the driver's `.so`. The loader enumerates every manifest it finds and
   exposes every ICD as a separate Vulkan "physical device" — a box with both
   an AMD and an NVIDIA GPU installed shows both.
3. **The kernel driver** — already present on any modern Linux: `amdgpu` for
   AMD, `i915`/`xe` for Intel, `nvidia`/`nouveau` for NVIDIA. It exposes
   `/dev/dri/renderD1XX`, which is all a Vulkan *compute*-only workload (no
   display) needs to touch.

`libshaderc1` (Shaderc) is a separate, vendor-neutral library nablatensor also
dlopens: it compiles GLSL compute shaders to SPIR-V at runtime, so the fused
kernel nablatensor generates for a given tape is turned into SPIR-V on the fly
rather than shipped precompiled.

## Installing by vendor

### AMD (Mesa RADV) — open source, in every distro's repos

```bash
# Debian / Ubuntu
sudo apt-get install mesa-vulkan-drivers libvulkan1 libshaderc1 vulkan-tools

# Fedora / RHEL / CentOS Stream
sudo dnf install mesa-vulkan-drivers vulkan-loader libshaderc vulkan-tools

# Arch / Manjaro
sudo pacman -S vulkan-radeon vulkan-icd-loader shaderc vulkan-tools

# openSUSE
sudo zypper install libvulkan_radeon vulkan-loader libshaderc1 vulkan-tools
```

`mesa-vulkan-drivers` (Debian/Fedora naming) is a single package containing
**every** Mesa ICD — RADV (AMD), ANV (Intel), NVK (Nouveau/NVIDIA open driver),
and llvmpipe (CPU) — so installing it is safe even before you know which GPU
is in the machine; the loader only activates the ICD matching hardware it
finds. `vulkan-tools` is optional — it is only the `vulkaninfo` diagnostic, not
a runtime dependency.

No proprietary driver, firmware blob, or reboot is needed: the in-tree
`amdgpu` kernel module (present in any kernel from the last ~10 years) already
exposes `/dev/dri/renderD128` (or `renderD129`, `...` for a second GPU); RADV
runs entirely in userspace on top of it.

### Intel (Mesa ANV) — same package, same story

Nothing additional beyond the `mesa-vulkan-drivers` install above — ANV ships
in the same package as RADV. The kernel side is the in-tree `i915` driver (or
`xe` on very recent Intel Arc / Battlemage hardware, kernel 6.8+), again
already present.

### NVIDIA — proprietary driver, not Mesa

Do **not** rely on `mesa-vulkan-drivers` for an NVIDIA GPU with the proprietary
stack (Mesa's NVK, the open-source Nouveau-based ICD, is improving but is not
what you want for production compute). Install NVIDIA's own driver package
instead — it drops its own ICD manifest and `.so`:

```bash
# Debian / Ubuntu
sudo apt-get install nvidia-driver-<version>   # e.g. nvidia-driver-550
sudo apt-get install libvulkan1 vulkan-tools   # loader + diagnostics (vendor-neutral)

# Fedora (via RPM Fusion) / Arch
sudo dnf install akmod-nvidia vulkan-loader vulkan-tools
sudo pacman -S nvidia vulkan-icd-loader vulkan-tools
```

The NVIDIA installer places its ICD manifest at
`/usr/share/vulkan/icd.d/nvidia_icd.json` (Linux) and needs the matching
kernel module (`nvidia.ko`) loaded — check with `nvidia-smi`. A reboot (or at
least a driver/module reload) is typically required after install, unlike the
Mesa case. `libshaderc1`/`libshaderc` is still needed separately — NVIDIA's
package does not provide it.

### No GPU at all — software Vulkan (lavapipe)

For CI, containers with no `/dev/dri` device, or plain development boxes,
Mesa's `llvmpipe`/**lavapipe** ICD runs Vulkan (including compute) entirely on
the CPU. It ships in the same `mesa-vulkan-drivers` package; force the loader
to pick it (skipping any real GPU ICD also present) with:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json java ...
```

It is slow relative to any real GPU or even nablatensor's own `simd`/`cpu-jit`
backends, but it is useful for verifying that Vulkan dispatch code paths are
exercised at all in an environment where no accelerator is available.

## Verifying the install

```bash
vulkaninfo --summary
```

lists every enumerated physical device with its `deviceName`, `driverName`,
`driverInfo` and `apiVersion`. Nothing listed means either no ICD manifest was
found (check `/usr/share/vulkan/icd.d/`) or the loader itself is missing
(`ldconfig -p | grep libvulkan`).

## Picking a specific device when more than one is present

If the loader enumerates multiple physical devices (e.g. an iGPU and a
discrete GPU, or a real GPU alongside lavapipe) and you need to pin one:

```bash
# Point the loader at exactly one ICD manifest, bypassing the others entirely
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json java ...

# Or, with newer loaders, filter by driver name without touching ICD files
VK_LOADER_DRIVERS_SELECT=radv java ...

# Or select by device index once all ICDs are enumerated (Vulkan >=1.3 loaders)
VK_DEVICE_SELECT=list java ...   # then VK_DEVICE_SELECT=<index>
```

nablatensor itself does not currently expose a device-index selector of its
own — it takes whatever physical device index 0 the loader hands back — so use
these loader-level environment variables to steer which GPU is used on a
multi-GPU host.

## Running inside a container

Vulkan compute inside a container needs three things, regardless of which
container runtime (Podman, Docker, ...) or vendor is involved:

1. **The userspace pieces installed in the image** — the loader, an ICD
   matching the *host's* GPU vendor, and `libshaderc`. For NVIDIA specifically,
   this typically means using the vendor's own container toolkit
   (`nvidia-container-toolkit`) rather than trying to manually copy the host's
   proprietary driver into the image, since that driver is version-locked to
   the host kernel module.
2. **The device node passed through** — `--device /dev/dri/renderD128` (adjust
   the number if more than one GPU is present; find yours with
   `ls /dev/dri/`). This is the same node Vulkan and ROCm both use for
   compute — no separate "Vulkan device" exists at the kernel level.
3. **Group membership preserved** — the render node is normally owned by group
   `render`, so the containerized process needs to run as, or keep, a group
   that has access to it (e.g. Podman's `--group-add keep-groups`, or matching
   `--group-add <gid>` to the host's `render` GID under Docker).

`container/Containerfile` and `container/run-demo.sh` in this repository are a
worked example of steps 1–3 for the Mesa RADV (AMD) case; swap the `apt-get
install` line for the NVIDIA container toolkit's setup if targeting an NVIDIA
host instead.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `vulkaninfo` reports "Cannot create Instance" or lists nothing | No ICD manifest under `/usr/share/vulkan/icd.d/`; driver package not installed for this vendor |
| Device visible to `vulkaninfo` on the host but not inside a container | `/dev/dri/renderD1XX` not passed through, or the container process isn't in the group that owns it |
| nablatensor's `vulkan` backend reports itself unavailable (`isAvailable() == false`) at runtime | `libvulkan.so.1` or `libshaderc.so.1` not on the library search path — check `ldconfig -p \| grep -E 'libvulkan\|libshaderc'` |
| Wrong GPU picked on a multi-GPU host | Use `VK_ICD_FILENAMES` / `VK_LOADER_DRIVERS_SELECT` / `VK_DEVICE_SELECT` as above |
| Works as root, fails as a normal user | User not in the `render` (and for display use, `video`) group; `sudo usermod -aG render $USER` then re-login |
