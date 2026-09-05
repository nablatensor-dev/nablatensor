# Incident: running the `nablatensor-bench` benchmark crashed the desktop session

**Date:** 2026-09-05, ~09:29:46–09:30:17 local time
**Machine:** `nucbox` (GMKtec NucBox K8 Plus), AMD Ryzen 7 8845HS, integrated
Radeon 780M (`gfx1103`, PCI `Phoenix3 [1002:1900]`), kernel `7.0.0-30-generic`,
ROCm `7.2.4` installed at `/opt/rocm-7.2.4`.
**Trigger:** `mvn -o -q -pl nablatensor-bench exec:java -Dexec.mainClass=com.nablatensor.bench.Benchmarks -Dscenarios=2000000 -Dsteps=252`

## Summary

Running the repo's own `Benchmarks` harness knocked out the GNOME desktop
session on this machine. The kernel itself did not reboot (`uptime -s` shows
the same boot as before the run), but the GPU driver hit an unrecoverable
command-queue fault, performed an internal GPU reset, and the display
controller reinit that followed hit its own bug — which took the desktop
compositor down with it. From the user's chair this reads exactly like "the
PC crashed": screen disruption, GDM login screen reappearing, session
services failing to restart.

**Root cause, in one sentence:** the benchmark's `backendMatrix()` step
unconditionally tries every *discovered* AAD engine — including `rocm` — on a
machine where the only GPU is an integrated, consumer-class APU (`gfx1103`)
that is simultaneously driving the live display; the ROCm kernel launch
faulted, and recovering the GPU from that fault destabilized the display
pipeline sharing the same hardware.

## Evidence

`uptime` proves no reboot occurred:

```
$ uptime -s
2026-08-31 13:20:16
```

`journalctl -k -b` for the incident window (only the relevant lines):

```
09:29:46 amdgpu 0000:c6:00.0: MES failed to respond to msg=REMOVE_QUEUE
09:29:46 amdgpu 0000:c6:00.0: failed to remove hardware queue from MES, doorbell=0x1002
09:29:46 amdgpu 0000:c6:00.0: MES might be in unrecoverable state, issue a GPU reset
09:29:46 amdgpu 0000:c6:00.0: Failed to evict queue 1
09:29:46 amdgpu 0000:c6:00.0: Failed to evict process queues
09:29:46 amdgpu 0000:c6:00.0: remove_all_kfd_queues_mes: Failed to remove queue 0 for dev 20592
09:29:49 amdgpu 0000:c6:00.0: MES failed to respond to msg=REMOVE_QUEUE
09:29:49 amdgpu 0000:c6:00.0: failed to unmap legacy queue
09:29:51 amdgpu 0000:c6:00.0: MES failed to respond to msg=REMOVE_QUEUE
09:29:51 amdgpu 0000:c6:00.0: failed to unmap legacy queue
09:29:53 amdgpu 0000:c6:00.0: MES failed to respond to msg=REMOVE_QUEUE
09:29:53 amdgpu 0000:c6:00.0: failed to unmap legacy queue
09:29:53 [drm:gfx_v11_0_cp_gfx_enable.isra.0 [amdgpu]] *ERROR* failed to halt cp gfx
09:30:03 amdgpu 0000:c6:00.0: MES failed to respond to msg=REMOVE_QUEUE
09:30:03 amdgpu 0000:c6:00.0: failed to remove hardware queue from MES, doorbell=0x1000
09:30:03 amdgpu 0000:c6:00.0: MES might be in unrecoverable state, issue a GPU reset
09:30:03 amdgpu: Pasid 0x352 destroy queue 0 failed, ret -110
09:30:03 amdgpu 0000:c6:00.0: [drm] AMDGPU device coredump file has been created
09:30:03 amdgpu 0000:c6:00.0: SMU is resuming...
09:30:03 amdgpu 0000:c6:00.0: SMU is resumed successfully!
09:30:03 amdgpu 0000:c6:00.0: GPU reset(2) succeeded!
09:30:03 amdgpu 0000:c6:00.0: [drm] device wedged, but recovered through reset
09:30:05 amdgpu 0000:c6:00.0: [drm] REG_WAIT timeout 1us * 100 tries - dcn31_program_compbuf_size line:142
09:30:05 WARNING: .../dcn31/dcn31_hubbub.c:151 at dcn31_program_compbuf_size+0xce/0x230 [amdgpu]
         Call Trace: dcn20_optimize_bandwidth -> dc_commit_state_no_check -> dc_commit_streams
                      -> amdgpu_dm_commit_streams -> amdgpu_dm_atomic_commit_tail
                      -> commit_tail -> drm_atomic_helper_commit -> drm_atomic_commit
                      -> atomic_remove_fb -> drm_framebuffer_remove -> drm_mode_rmfb_work_fn
09:30:05 Failed to start org.gnome.SettingsDaemon.Keyboard.service
09:30:05 Failed to start org.gnome.SettingsDaemon.Color.service
09:30:05 Failed to start org.gnome.SettingsDaemon.MediaKeys.service
09:30:06 gdm3: Gdm: on_display_added: assertion 'GDM_IS_REMOTE_DISPLAY (display)' failed
09:30:06 gdm3: Gdm: on_display_removed: assertion 'GDM_IS_REMOTE_DISPLAY (display)' failed
09:30:15 gdm-password: gkr-pam: unable to locate daemon control file
09:30:15 Failed to start app-gnome-gnome-keyring-secrets-....scope
09:30:16 Failed to start app-gnome-spice-vdagent-....scope
```

This is a single, continuous causal chain, not two unrelated problems:

1. **09:29:46–09:30:03** — the GPU's command scheduler (MES firmware) stops
   responding to queue-teardown requests. `amdgpu` eventually gives up and
   issues its own internal GPU reset (`GPU reset(2) succeeded!`), which is the
   driver's last-resort recovery for a wedged device.
2. **09:30:03–09:30:05** — while the display controller reinitializes after
   that reset, it hits a firmware/driver bug of its own
   (`REG_WAIT timeout ... dcn31_program_compbuf_size`), producing a kernel
   `WARNING` in the atomic-commit path that's actually rendering the desktop.
3. **09:30:05 onward** — the GNOME session's own services start failing to
   (re)start, and GDM's assertions fire as it re-attaches a display — this is
   the "screen came back to a login prompt" moment the user experienced.

The correlating Java-side output from the same run (captured before the crash
interrupted the session):

```
## backend matrix  (one tape, same seed)

| engine | price | delta | scenarios/s | runs on |
|---|--:|--:|--:|---|
| rocm | — | — | — | skipped: hipDeviceSynchronize failed: unspecified launch failure (code 719) |
| cpu | 5.301676 | 0.561932 | 8.91e+05 | scalar JVM · 16 processors · fp64 |
| cpu-jit | 5.301676 | 0.561932 | 1.59e+06 | generated straight-line bytecode kernel · ... |
```

`ProductBench.java`'s `backendMatrix()`/`Benchmarks.java`'s `backendMatrix()`
catch the HIP failure as a plain `RuntimeException` and print "skipped" —
which is exactly correct at the JVM level and is why `mvn`/the JVM itself
never crashed. But that catch block hides how serious the underlying event
was: by the time `hipDeviceSynchronize` throws, the GPU firmware is already
in the process of wedging, and the kernel-level recovery from that is what
actually disrupts the machine.

## Why this GPU is unusually exposed

```
$ lspci -nn | grep -i vga
c6:00.0 VGA compatible controller [0300]: AMD/ATI Phoenix3 [1002:1900] (rev c5)

$ rocminfo | grep -E "Name:|Marketing"
Name:            AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
Name:            gfx1103
Marketing Name:  AMD Radeon Graphics
```

- There is **one GPU** in this machine, an integrated `gfx1103` APU part. It
  is simultaneously the only display adapter *and* the only ROCm/HIP compute
  device — unlike a workstation with a discrete compute card, there is no
  isolation between "the GPU running the benchmark" and "the GPU rendering
  the desktop."
- `gfx1103` (Radeon 780M / "Phoenix") is a consumer laptop/mini-PC APU
  target. It is not among AMD's officially validated ROCm GPUs (that list is
  centered on discrete data-center/workstation parts); support here is
  best-effort, and is known within this repo to be less stable — see the
  existing comment in `RocmAadEngine`:

  > "on this repo's AMD-APU dev box `Nabla.model(...).fastest()` keeps
  > choosing Vulkan (the RADV compute path has proven the more reliable of
  > the two here); ROCm is selected explicitly ... which is the intended way
  > to benchmark it."

  (`nablatensor-rocm/src/main/java/com/nablatensor/engine/rocm/RocmAadEngine.java`)

  In other words: the project already knows ROCm is the less reliable
  backend on exactly this hardware class, and already keeps its `.fastest()`
  priority (55) below Vulkan for that reason. `backendMatrix()` bypasses that
  safeguard entirely, since it iterates every *discovered/available* engine
  rather than going through `.fastest()`.

## Root cause chain

1. `Benchmarks.main()` calls `backendMatrix()`, which loops over
   `AadEngines.available(options)` and actually builds + runs a `MonteCarlo`
   on **every** engine that reports itself available — with no opt-in flag,
   no allowlist, and no isolation from the interactive session.
2. `RocmAadEngine.isAvailable()` only checks `HipCompute.isAvailable()` —
   i.e. "is `libamdhip64` loadable and does it see a device" — not whether
   the device is one ROCm is known to run compute kernels on reliably.
   (`nablatensor-rocm/.../RocmAadEngine.java:48-53`)
3. Because ROCm 7.2.4 is installed and this APU is HSA-visible,
   `isAvailable()` returns `true`, so the harness compiles and launches a
   real fused forward+adjoint HIP kernel via HIPRTC on the same GPU driving
   the desktop.
4. The kernel launch faults. The HIP runtime reports this cleanly to the JVM
   (`hipDeviceSynchronize failed: unspecified launch failure`, code 719) —
   this part works as designed.
5. At the driver/firmware level, though, the GPU's command scheduler is left
   wedged. `amdgpu` notices over the next ~17 seconds (repeated
   `MES failed to respond to msg=REMOVE_QUEUE`) and performs an internal GPU
   reset to recover it.
6. The display-controller reinitialization that reset triggers hits its own
   kernel bug (`dcn31_program_compbuf_size` `REG_WAIT` timeout), and that is
   what actually disrupts the live GNOME session — because, again, it's the
   same physical GPU serving both roles.

## Recommended fixes

1. **Don't let a full-sweep benchmark (`backendMatrix()` in both
   `Benchmarks.java` and `ProductBench.java`) auto-probe every discovered
   engine by default.** Gate GPU backends behind an explicit opt-in (e.g. a
   `-Dnablatensor.bench.gpu=true` system property, off by default), so
   running the plain benchmark command never touches a GPU unless asked.
2. **Give `RocmAadEngine` (and the other GPU engines) a real capability
   check, not just a presence check.** A minimal no-op kernel
   launch-and-synchronize "smoke test," on a short timeout, run once and
   cached, would catch this exact failure mode *before* `available()`
   reports the engine usable — instead of finding out via a real
   forward+adjoint kernel launch inside a user's benchmark run.
3. **Consider a documented denylist/allowlist by `gfx` target** for the ROCm
   engine (`gfx1103` and other consumer RDNA3 APU targets specifically),
   consistent with the fact that this repo's own priority ordering already
   treats this hardware class as second-best-effort.
4. **Document the shared-display-GPU risk explicitly** wherever the ROCm/GPU
   backends are discussed (README, `docs/`, `nablatensor-bench`'s own
   Javadoc): running a GPU compute backend on a machine where that same GPU
   drives the interactive desktop can crash/restart the session if the
   compute kernel faults, and such benchmarking should be done from a
   non-interactive shell (SSH from another machine, a TTY without a running
   desktop session, or a headless/server box) rather than a local desktop
   terminal.
5. **Make a faulted-engine failure louder in the harness's own output.** The
   current "skipped: hipDeviceSynchronize failed..." line reads as a benign,
   fully contained failure. At minimum it should note that a GPU fault can
   have already destabilized the system by the time this message prints.

## Files involved

- `nablatensor-bench/src/main/java/com/nablatensor/bench/Benchmarks.java` —
  `backendMatrix()`, unconditional engine sweep.
- `nablatensor-bench/src/main/java/com/nablatensor/bench/ProductBench.java` —
  same pattern, per-product.
- `nablatensor-core/.../AadEngines.java` — `available()`/`discovered()`,
  the generic engine-enumeration logic that both harnesses rely on.
- `nablatensor-rocm/src/main/java/com/nablatensor/engine/rocm/RocmAadEngine.java`
  — `isAvailable()`, the presence-only check that should become a real
  capability probe.
