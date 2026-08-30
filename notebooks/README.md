# `notebooks/` — NablaTensor from Jupyter

| Notebook | Mirrors | What you run |
|---|---|---|
| [`greeks-on-gpu.ipynb`](greeks-on-gpu.ipynb) | [`demo/greeks-on-gpu.sh`](../demo/greeks-on-gpu.sh) | A down-and-in put written as a plain-Java lambda, recorded once, then **20 M paths priced with price + delta + vega + rho + dV/dK + dV/dT from one reverse sweep** — plus a spot ladder (with a plot) and a crash scenario on the same compiled kernel. |

The notebook drives the **same Java code** as the shell demo. A small package,
[`python/nablatensor`](../python/nablatensor), boots one JDK 25 JVM through
[JPype](https://jpype.readthedocs.io) against the checkout's compiled
`*/target/classes` and forwards every call to `com.nablatensor.quant.*`.

## Setup (venv)

Run these yourself from the **repo root** (`nablatensor/`).

### 1. Build the Java side — JDK 25 (the project's LTS baseline)

```sh
mvn -o compile
```

JPype needs the same JDK. It is picked from `$NABLATENSOR_JDK`, then `$JAVA_HOME`,
then the default `/opt/zulu25.30.17-ca-jdk25.0.1-linux_x64`. Point at least one
of those at a JDK 25 home (24+ also works — the generated-bytecode engine needs
the Class-File API):

```sh
export NABLATENSOR_JDK=/opt/zulu25.30.17-ca-jdk25.0.1-linux_x64
```

### 2. Create the virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r notebooks/requirements.txt
```

That installs the local `python/` bridge in editable mode plus JupyterLab,
ipykernel and matplotlib (see [`notebooks/requirements.txt`](requirements.txt)).

### 3. Register the kernel (optional, but keeps the venv selectable)

```sh
python -m ipykernel install --user --name nablatensor \
  --display-name "Python (nablatensor)"
```

### 4. Launch

```sh
source .venv/bin/activate
export NABLATENSOR_JDK=/opt/zulu25.30.17-ca-jdk25.0.1-linux_x64
jupyter lab notebooks/greeks-on-gpu.ipynb
```

Pick the **Python (nablatensor)** kernel (or the `.venv` interpreter) and run
all cells.

## Notes

- **`WARNING: Using incubator modules: jdk.incubator.vector`** on the first cell
  is expected — the `simd` engine is the JDK Vector API.
- The JVM starts **once per kernel process**. After any `mvn -o compile`,
  restart the kernel (*Kernel ▸ Restart*) before the new classes are visible.
- **Backend.** `ENGINE = nt.best_engine()` picks the fastest fp32 adjoint
  backend this machine has — `vulkan`, else `rocm`, else the pure-Java
  `cpu-jit`. Force one by editing that line, e.g. `ENGINE = "cpu-jit"`.
  `nt.engines()` lists what the ServiceLoader found and whether each is usable.
- **Non-default checkout / JDK.**
  `nt.start(project_root="/path/to/nablatensor", jdk_home="/path/to/jdk")`.
- The bridge is a convenience for these examples, not a supported public API —
  see [`python/README.md`](../python/README.md).
