# CUDA-Q Reference Plugins

Two example external QPU plugins showing how to extend CUDA-Q with custom
backends without modifying the core tree.

| Plugin | Shape | What it demonstrates |
|--------|-------|---------------------|
| `mock_rest/` | REST | `ServerHelper` subclass, auto-loaded via `platform-qpu: remote_rest` |
| `trace_qpu/` | Full QPU | `QPU` subclass + custom MLIR pass (`trace-qpu-summary`), loaded via `plugin-libraries` |

## Build

Each plugin is a standalone CMake project. Add the examples explicitly with
`CUDAQ_EXTERNAL_PROJECTS`; they are not part of the default CUDA-Q build.

```sh
cmake -B build \
  -DCUDAQ_EXTERNAL_PROJECTS="mock-rest;trace-qpu" \
  -DCUDAQ_EXTERNAL_MOCK_REST_SOURCE_DIR=$PWD/examples/plugins/mock_rest \
  -DCUDAQ_EXTERNAL_TRACE_QPU_SOURCE_DIR=$PWD/examples/plugins/trace_qpu

ninja -C build cudaq-example-serverhelper-mock_rest cudaq-example-qpu-trace_qpu
```

Outputs land in `build/external/<name>/`. Each output directory contains the
plugin `targets/`, `lib/`, and Python packaging metadata.

## Install

For Python discovery, install the built plugin root as a Python package:

```sh
python3 -m pip install build/external/trace-qpu
```

After that, `import cudaq` loads the plugin's `cudaq.backends` entry point and
`cudaq.set_target("trace_qpu")` works without an explicit registration call.

To also make the same package visible to `nvq++`, use the package's helper:

```sh
python3 -m cudaq_example_trace_qpu --install-nvqpp
```

For C++-only workflows, install the plugin root directly:

```sh
cudaq-install-plugin build/external/trace-qpu   # visible to nvq++
```

During development, you can still register the build output directly:

```python
import cudaq
cudaq.register_backend_path("build/external/trace-qpu")
cudaq.set_target("trace_qpu")
```

## Test

After configuring the plugins with `CUDAQ_EXTERNAL_PROJECTS`, run their explicit
check targets:

```sh
ninja -C build check-mock-rest check-trace-qpu
```

## Create your own

Copy either directory, rename the YAML/sources/targets, and build with:

```sh
cmake -B build \
  -DCUDAQ_EXTERNAL_PROJECTS="my-plugin" \
  -DCUDAQ_EXTERNAL_MY_PLUGIN_SOURCE_DIR=/path/to/my-plugin
```
