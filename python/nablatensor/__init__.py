"""A thin Python view onto the NablaTensor quant / adjoint API, for notebooks.

This is *not* a reimplementation of anything. It boots a single JDK 25 JVM
through `JPype <https://jpype.readthedocs.io>`_ against the compiled
``*/target/classes`` of a NablaTensor checkout and re-exports the handful of
Java types the ``notebooks/`` need:

    >>> import nablatensor as nt
    >>> nt.start()
    >>> market = nt.EquityMarket(100.0, 100.0, 0.28, 0.03, 1.0)
    >>> note   = nt.ExoticProducts.barrier(
    ...     nt.OptionType.PUT, nt.ExoticProducts.Barrier.DOWN_IN, 70.0, 1.0)
    >>> mc = (nt.MonteCarlo.of(note).market(market).steps(252)
    ...         .fp32().greeks().on(nt.best_engine()).build())
    >>> p = mc.run(20_000_000, 42)
    >>> p.price(), p.delta(), p.vega()

Every method on the returned objects is the real Java method (JPype forwards
the call), so the demo scripts in ``demo/*.sh`` and these notebooks run the
exact same code paths.

The JVM can only be started once per process: after ``mvn -o compile`` you must
restart the notebook kernel before the new classes are visible.

Built and run against JDK 25 (the project's LTS baseline).
"""

from __future__ import annotations

import glob
import os
from typing import Any

__all__ = [
    "start",
    "started",
    "engines",
    "best_engine",
    "jclass",
    "EquityMarket",
    "OptionType",
    "ExoticProducts",
    "Products",
    "MonteCarlo",
    "Pricing",
    "AadEngines",
    "AadEngine",
]

# The JDK NablaTensor is built with. Overridden by $NABLATENSOR_JDK, then
# $JAVA_HOME, then this default. JDK 25 is the project's LTS baseline (pom.xml
# sets maven.compiler.release=25; CI runs on 25). JDK 24+ also works — the
# generated-bytecode engine needs the Class-File API.
_DEFAULT_JDK_HOME = "/opt/zulu25.30.17-ca-jdk25.0.1-linux_x64"

# Fully qualified names of the Java types re-exported as module attributes.
_EXPORTS = {
    "EquityMarket": "com.nablatensor.quant.EquityMarket",
    "OptionType": "com.nablatensor.quant.OptionType",
    "ExoticProducts": "com.nablatensor.quant.ExoticProducts",
    "Products": "com.nablatensor.quant.Products",
    "MonteCarlo": "com.nablatensor.quant.MonteCarlo",
    "Pricing": "com.nablatensor.quant.Pricing",
    "AadEngines": "com.nablatensor.engine.AadEngines",
    "AadEngine": "com.nablatensor.engine.AadEngine",
}

_started = False


def _find_project_root() -> str:
    """Walk up from this file looking for the Maven reactor ``pom.xml``."""
    here = os.path.dirname(os.path.abspath(__file__))
    # python/nablatensor/__init__.py -> the checkout root is two levels up, but
    # confirm it by the pom rather than trusting the layout.
    candidate = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    directory = candidate
    while True:
        if os.path.isfile(os.path.join(directory, "pom.xml")):
            return directory
        parent = os.path.dirname(directory)
        if parent == directory:
            return candidate
        directory = parent


def _jdk_home(explicit: str | None) -> str:
    return (
        explicit
        or os.environ.get("NABLATENSOR_JDK")
        or os.environ.get("JAVA_HOME")
        or _DEFAULT_JDK_HOME
    )


def start(
    project_root: str | None = None,
    jdk_home: str | None = None,
    engine: str | None = None,
    extra_jvm_args: tuple[str, ...] = (),
) -> None:
    """Start the JVM and wire the NablaTensor classpath. Idempotent.

    Parameters
    ----------
    project_root:
        A NablaTensor checkout. Defaults to the one containing this file.
    jdk_home:
        JDK 25 (or 24+) home for JPype. Defaults to ``$NABLATENSOR_JDK`` /
        ``$JAVA_HOME`` / a Zulu 25 path.
    engine:
        If given, pins the adjoint backend for the whole session via the
        ``nablatensor.engine`` system property (``cpu``, ``cpu-jit``, ``simd``,
        ``rocm``, ``vulkan``, ``cuda``). Usually you leave this ``None`` and pass
        an engine per build instead — see :func:`best_engine`.
    extra_jvm_args:
        Extra raw arguments forwarded to ``jpype.startJVM``.
    """
    global _started
    if _started:
        return

    import jpype
    import jpype.imports  # noqa: F401  (enables `import com.nablatensor...`)

    root = project_root or _find_project_root()
    classpath = sorted(glob.glob(os.path.join(root, "*", "target", "classes")))
    if not classpath:
        raise RuntimeError(
            f"no compiled classes under {root}/*/target/classes — "
            f"run `mvn -o compile` in the checkout first"
        )

    home = _jdk_home(jdk_home)
    libjvm = os.path.join(home, "lib", "server", "libjvm.so")
    if not os.path.isfile(libjvm):
        raise RuntimeError(
            f"no libjvm.so at {libjvm} — set NABLATENSOR_JDK to a JDK 24+ home"
        )

    args = [
        # FFI backends (rocm/vulkan/cuda) call restricted native methods.
        "--enable-native-access=ALL-UNNAMED",
        # the `simd` engine is the JDK Vector API, still an incubator module.
        "--add-modules=jdk.incubator.vector",
    ]
    if engine:
        args.append(f"-Dnablatensor.engine={engine}")
    args.extend(extra_jvm_args)

    jpype.startJVM(libjvm, *args, classpath=classpath)
    _started = True


def started() -> bool:
    """Whether :func:`start` has run in this process."""
    return _started


def _ensure_started() -> None:
    if not _started:
        start()


def jclass(name: str) -> Any:
    """``jpype.JClass(name)`` after ensuring the JVM is up.

    An escape hatch for Java types this module does not re-export by name, e.g.
    ``nt.jclass("com.nablatensor.quant.Products").asianCall()``.
    """
    _ensure_started()
    import jpype

    return jpype.JClass(name)


def engines() -> list[dict[str, Any]]:
    """Every adjoint engine the ServiceLoader finds, highest priority first.

    Each entry is ``{"name", "priority", "available", "describe"}``; mirrors
    ``AadEngines.discovered()`` in ``demo/_player.sh``'s probe.
    """
    _ensure_started()
    discovered = jclass("com.nablatensor.engine.AadEngines").discovered()
    return [
        {
            "name": str(e.name()),
            "priority": int(e.priority()),
            "available": bool(e.isAvailable()),
            "describe": str(e.describe()),
        }
        for e in discovered
    ]


def best_engine() -> str:
    """The engine ``demo/greeks-on-gpu.sh`` would pick on this machine.

    Fastest available fp32 accelerator first — ``vulkan`` then ``rocm`` — then
    the pure-Java ``cpu-jit`` fallback, then whatever else reports available.
    """
    available = {e["name"] for e in engines() if e["available"]}
    for preferred in ("vulkan", "rocm", "cpu-jit"):
        if preferred in available:
            return preferred
    for e in engines():
        if e["available"]:
            return e["name"]
    return "cpu-jit"


def __getattr__(name: str) -> Any:  # PEP 562 — lazy, JVM-backed module attrs
    fqn = _EXPORTS.get(name)
    if fqn is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return jclass(fqn)


def __dir__() -> list[str]:
    return sorted(__all__)
