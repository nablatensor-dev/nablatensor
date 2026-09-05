#!/usr/bin/env bash
# Run something inside the nablatensor demo-runner image, from this PC.
#
#   container/run-demo.sh <engine> [command ...]
#
#   <engine>   cpu-jit | vulkan | rocm | -      (- = let the probe pick)
#   command    what to run in the container; default: an interactive bash
#
# `container/build.sh` builds a CPU-only image, so only `cpu-jit` works
# against it out of the box. `vulkan` / `rocm` need an image with that GPU
# stack installed — see docs/install/gpu-container.md for how to build one
# (this script's device-passthrough flags below apply unchanged to that
# image; only $NT_IMAGE needs to point at it, e.g. `NT_IMAGE=... run-demo.sh
# vulkan ...`).
#
# Examples:
#   container/run-demo.sh cpu-jit ./demo/isda-simm-full.sh --fast
#   container/run-demo.sh vulkan  nt-gpucheck     # against a GPU-enabled image
#   container/run-demo.sh rocm    ./demo/one-tape-every-backend.sh --fast
#
# GPU access is rootless podman device passthrough:
#   vulkan -> /dev/dri/renderD128
#   rocm   -> /dev/dri/renderD128 + /dev/kfd   (+ seccomp:unconfined for HSA)
# with --group-add keep-groups so the container process keeps the host's
# `render` / `video` groups that own those device nodes.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$here/_env.sh"

engine=${1:--}; shift || true

common=( --rm --init --group-add keep-groups
         -e HOME=/root -e "TERM=${TERM:-xterm}" -w /opt/nablatensor )
[ -t 0 ] && [ -t 1 ] && common+=( -it )

case "$engine" in
  cpu-jit) extra=( -e NABLATENSOR_DEMO_ENGINE=cpu-jit ) ;;
  vulkan)  extra=( -e NABLATENSOR_DEMO_ENGINE=vulkan
                   --device /dev/dri/renderD128 ) ;;
  rocm)    extra=( -e NABLATENSOR_DEMO_ENGINE=rocm
                   --device /dev/dri/renderD128 --device /dev/kfd
                   --security-opt seccomp=unconfined ) ;;
  -|'')    extra=( --device /dev/dri/renderD128 --device /dev/kfd
                   --security-opt seccomp=unconfined ) ;;
  *) echo "usage: run-demo.sh <cpu-jit|vulkan|rocm|-> [command ...]" >&2; exit 2 ;;
esac

set -- "${@:-bash}"
exec "${PODMAN[@]}" run "${common[@]}" "${extra[@]}" "$NT_IMAGE" "$@"
