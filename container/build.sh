#!/usr/bin/env bash
# Build the nablatensor demo-runner image with podman.
#
#   container/build.sh
#
# Needs, on the host:
#   * a JDK 26 directory     (default /opt/zulu26.32.13-ca-jdk26.0.2-linux_x64,
#                             override with $NABLATENSOR_JDK)
#   * a working ROCm install (default /opt/rocm, override with $ROCM_PATH)
#   * the project compiled    (this script runs `mvn -o compile` if it isn't)
#
# The host /opt/rocm is ~22 GB; only ~185 MB of it is needed, so this script
# stages just the HIP runtime + runtime compiler + device bitcode into a temp
# dir and hands that to `podman build` as the `rocm` build context. The JDK is
# handed in the same way as the `jdk` context.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/.." && pwd)
. "$here/_env.sh"

JDK_DIR=${NABLATENSOR_JDK:-/opt/zulu26.32.13-ca-jdk26.0.2-linux_x64}
ROCM_DIR=${ROCM_PATH:-/opt/rocm}
HOST_MVN_JDK=${JAVA_HOME:-/opt/zulu25.30.17-ca-jdk25.0.1-linux_x64}

[ -x "$JDK_DIR/bin/jshell" ]          || { echo "no JDK 26 at $JDK_DIR (set \$NABLATENSOR_JDK)" >&2; exit 1; }
[ -e "$ROCM_DIR/lib/libamdhip64.so" ] || { echo "no ROCm at $ROCM_DIR (set \$ROCM_PATH)"        >&2; exit 1; }
command -v fuse-overlayfs >/dev/null  || { echo "fuse-overlayfs not installed"                   >&2; exit 1; }

# bundled static crun — runc can't keep host groups, which the ROCm path needs
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

# 1 · stage the minimal ROCm userspace
stage=$(mktemp -d /tmp/nt-rocm-stage.XXXXXX)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/lib" "$stage/bin"
echo ">> staging minimal ROCm from $ROCM_DIR"
for so in libamdhip64 libhiprtc libhiprtc-builtins libamd_comgr \
          libhsa-runtime64 librocprofiler-register; do
  cp -aP "$ROCM_DIR"/lib/${so}.so* "$stage/lib/" 2>/dev/null \
    || cp -a "$(readlink -f "$ROCM_DIR/lib/${so}.so")" "$stage/lib/"
done
cp -a "$ROCM_DIR/amdgcn"                     "$stage/"     2>/dev/null || true
cp -a "$ROCM_DIR/.info"                      "$stage/"     2>/dev/null || true
cp -a "$ROCM_DIR/bin/rocminfo"               "$stage/bin/" 2>/dev/null || true
cp -a "$ROCM_DIR/bin/rocm_agent_enumerator"  "$stage/bin/" 2>/dev/null || true
echo "   staged $(du -sh "$stage" | cut -f1)"

# 2 · build (isolated fuse-overlayfs store — see _env.sh)
echo ">> podman build -> $NT_IMAGE   (store: $NT_PODMAN_ROOT)"
"${PODMAN[@]}" build \
  --build-context jdk="$JDK_DIR" \
  --build-context rocm="$stage" \
  --ignorefile "$here/.containerignore" \
  -t "$NT_IMAGE" \
  -f "$here/Containerfile" \
  "$repo"

echo
echo ">> built $NT_IMAGE"
"${PODMAN[@]}" run --rm "$NT_IMAGE" cat /opt/SIZES.txt
