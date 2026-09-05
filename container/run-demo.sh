#!/usr/bin/env bash
# Run something inside the nablatensor demo-runner image, from this PC.
#
#   container/run-demo.sh <engine> [command ...]
#
#   <engine>   cpu-jit | vulkan | rocm | -      (- = let the probe pick)
#   command    what to run in the container; default: an interactive bash
#
# Examples:
#   container/run-demo.sh -                       # shell; probe picks the engine
#   container/run-demo.sh vulkan  nt-gpucheck     # show the Vulkan / ROCm devices
#   container/run-demo.sh rocm    ./demo/one-tape-every-backend.sh --fast
#   container/run-demo.sh cpu-jit ./demo/isda-simm-full.sh --fast
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
