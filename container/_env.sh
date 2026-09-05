# Shared config for container/build.sh and container/run-demo.sh.
#
# This box's default rootless-podman store uses the kernel-native overlay driver
# with overlay.ignore_chown_errors, under which `apt` cannot install (it can't
# write the _apt-owned cache dirs). So the demo image lives in its own podman
# store that uses fuse-overlayfs, which represents multiple UIDs correctly.
# Nothing here touches the user's normal `podman` store or config.

NT_IMAGE=${NT_IMAGE:-nablatensor-demo:latest}
NT_PODMAN_ROOT=${NT_PODMAN_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/nablatensor-podman}
NT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$NT_PODMAN_ROOT"

PODMAN=( podman --root "$NT_PODMAN_ROOT"
         --storage-driver overlay
         --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs )

# This box ships only `runc`, which cannot keep the host's supplementary groups
# (`--group-add keep-groups`); without them a rootless container process can't
# open the `render`-group /dev/kfd, so the ROCm engine fails. A bundled static
# `crun` (container/.bin/crun) does support it. `build.sh` fetches it if missing.
if [ -x "$NT_DIR/.bin/crun" ]; then
  PODMAN+=( --runtime "$NT_DIR/.bin/crun" )
fi
