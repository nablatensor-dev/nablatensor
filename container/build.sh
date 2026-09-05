#!/usr/bin/env bash
# Build the nablatensor demo-runner image with podman.
#
#   container/build.sh
#
# Needs, on the host:
#   * a JDK 26 directory     (default /opt/zulu26.32.13-ca-jdk26.0.2-linux_x64,
#                             override with $NABLATENSOR_JDK)
#   * the project compiled    (this script runs `mvn -o compile` if it isn't)
#
# This builds the CPU-only image (the `cpu-jit` engine, no GPU driver needed).
# To add a GPU engine (`vulkan` / `rocm`), see docs/install/gpu-container.md —
# it walks through extending the Containerfile with a GPU stage and building
# that variant instead.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/.." && pwd)
. "$here/_env.sh"

JDK_DIR=${NABLATENSOR_JDK:-/opt/zulu26.32.13-ca-jdk26.0.2-linux_x64}
HOST_MVN_JDK=${JAVA_HOME:-/opt/zulu25.30.17-ca-jdk25.0.1-linux_x64}

[ -x "$JDK_DIR/bin/jshell" ]          || { echo "no JDK 26 at $JDK_DIR (set \$NABLATENSOR_JDK)" >&2; exit 1; }
command -v fuse-overlayfs >/dev/null  || { echo "fuse-overlayfs not installed"                   >&2; exit 1; }

# bundled static crun — runc can't keep host groups, which a ROCm run needs
# for /dev/kfd access (see docs/install/gpu-container.md); harmless to fetch
# even when only running the CPU-only image built here.
if [ ! -x "$here/.bin/crun" ]; then
  echo ">> fetching static crun -> container/.bin/crun"
  mkdir -p "$here/.bin"
  curl -fsSL -o "$here/.bin/crun" \
    https://github.com/containers/crun/releases/download/1.29.1/crun-1.29.1-linux-amd64
  chmod +x "$here/.bin/crun"
fi
. "$here/_env.sh"   # re-source now that .bin/crun exists, so $PODMAN picks it up

# 0 · the demos run against */target/classes — compile on the host if needed
if ! find "$repo" -type d -path '*/target/classes/com' | grep -q .; then
  echo ">> compiling project on the host (mvn -o compile)"
  ( cd "$repo" && JAVA_HOME="$HOST_MVN_JDK" mvn -o -q compile )
fi

# 1 · build (isolated fuse-overlayfs store — see _env.sh)
echo ">> podman build -> $NT_IMAGE   (store: $NT_PODMAN_ROOT)"
"${PODMAN[@]}" build \
  --build-context jdk="$JDK_DIR" \
  --ignorefile "$here/.containerignore" \
  -t "$NT_IMAGE" \
  -f "$here/Containerfile" \
  "$repo"

echo
echo ">> built $NT_IMAGE"
