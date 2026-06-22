# trace_qpu Plugin

`trace_qpu` is a reference full-QPU external plugin. It demonstrates how to add
a target that provides its own `cudaq::QPU` implementation and ships a custom
MLIR pass in the same plugin library.

This target is a lightweight simulator-style target. It records launches and
returns deterministic synthetic results, so it is useful as a compact example of
QPU registration, target configuration, plugin-library loading, and pass
registration.

## What It Demonstrates

- A standalone external plugin directory with `targets/`, `lib/`, and Python
  package metadata generated into `build/external/trace-qpu`.
- A custom `cudaq::QPU` registered under the target name `trace_qpu`.
- A custom MLIR pass registered as `trace-qpu-summary`.
- A target YAML file that loads the plugin library through `plugin-libraries`
  so the pass is available when the target pass pipeline is parsed.
- Python auto-discovery through the `cudaq.backends` entry point.
- Optional installation into the user plugin scope for `nvq++`.

## How It Works

The key files are:

- `TraceQPU.cpp`: implements the `cudaq::QPU` subclass. It parses the optional
  `trace_file` backend argument, appends launch records to that file, and
  returns deterministic results for `sample` and `observe`.
- `TraceQPUSummaryPass.cpp`: implements the `trace-qpu-summary` MLIR pass. The
  pass walks each function, counts `quake.*` operations and measurements, and
  prints a compact summary.
- `targets/trace_qpu.yml.in`: declares the `trace_qpu` target, selects
  `platform-qpu: trace_qpu`, adds `func.func(trace-qpu-summary)` to the
  high-level JIT pipeline, and lists the plugin library so CUDA-Q loads the pass
  before parsing the pipeline.
- `python/__init__.py.in`: registers the built plugin root with CUDA-Q when the
  package is discovered through the `cudaq.backends` entry point.
- `python/__main__.py.in`: exposes `python3 -m cudaq_example_trace_qpu
  --install-nvqpp`, which installs the built plugin into the user plugin scope.

When a kernel is compiled for this target, the pass prints a summary like:

```text
trace-qpu-summary: kernel=... quake_ops=... measurements=...
```

When a kernel is launched, the QPU implementation can append a line like this to
the configured trace file:

```text
kernel=... context=sample shots=4
```

## Build

From the CUDA-Q repository root:

```sh
cmake -B build \
  -DCUDAQ_EXTERNAL_PROJECTS="trace-qpu" \
  -DCUDAQ_EXTERNAL_TRACE_QPU_SOURCE_DIR=$PWD/examples/plugins/trace_qpu

ninja -C build cudaq-example-qpu-trace_qpu
```

The built plugin root is:

```text
build/external/trace-qpu
```

It contains:

- `targets/trace_qpu.yml`
- `lib/libcudaq-qpu-trace_qpu.so` on Linux, or the platform equivalent
- `pyproject.toml`, `__init__.py`, and `__main__.py` for Python packaging

## Test

Run the plugin lit test:

```sh
ninja -C build check-trace-qpu
```

The test checks that the plugin build output exists, the YAML target is valid,
the custom pass can be loaded by `cudaq-opt`, the plugin can be installed into a
user plugin scope, `nvq++ --list-targets` finds it, and Python can execute a
sample kernel through direct registration and package entry-point discovery.

## Run

During development, register the build output directly:

```python
import cudaq

cudaq.register_backend_path("build/external/trace-qpu")
cudaq.set_target("trace_qpu", trace_file="trace_qpu.log")

kernel = cudaq.make_kernel()
q = kernel.qalloc()
kernel.h(q)
kernel.mz(q)

counts = cudaq.sample(kernel, shots_count=4)
counts.dump()

cudaq.reset_target()
```

The target returns deterministic all-zero counts for sampling. If `trace_file`
is set, each launch appends a record to that file.

## Install

For Python auto-discovery:

```sh
python3 -m pip install build/external/trace-qpu
```

After installation, `import cudaq` discovers the package's `cudaq.backends`
entry point, so explicit `cudaq.register_backend_path(...)` is no longer needed.

To make the same built package visible to `nvq++`:

```sh
python3 -m cudaq_example_trace_qpu --install-nvqpp
```

For C++-only workflows, install the built plugin root directly:

```sh
cudaq-install-plugin build/external/trace-qpu
```
