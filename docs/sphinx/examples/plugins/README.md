# CUDA-Q Reference Plugins

Two example external QPU plugins showing how to extend CUDA-Q with custom
backends without modifying the core tree.

| Plugin | Shape | What it demonstrates |
|--------|-------|---------------------|
| [`mock_rest/`](mock_rest/README.md) | REST | `ServerHelper` subclass for the built-in `remote_rest` QPU |
| [`trace_qpu/`](trace_qpu/README.md) | Full QPU | `QPU` subclass plus a custom MLIR pass |

## Build

Each plugin can be built as a standalone external project. Add the examples
explicitly with `CUDAQ_EXTERNAL_PROJECTS`; they are not part of the default
CUDA-Q build.

```sh
cmake -B build \
  -DCUDAQ_EXTERNAL_PROJECTS="mock-rest;trace-qpu" \
  -DCUDAQ_EXTERNAL_MOCK_REST_SOURCE_DIR=$PWD/examples/plugins/mock_rest \
  -DCUDAQ_EXTERNAL_TRACE_QPU_SOURCE_DIR=$PWD/examples/plugins/trace_qpu

ninja -C build cudaq-example-serverhelper-mock_rest cudaq-example-qpu-trace_qpu
```

Outputs land in `build/external/<name>/`. Each output directory contains the
plugin `targets/`, `lib/`, and Python packaging metadata.

## Test

After configuring the plugins with `CUDAQ_EXTERNAL_PROJECTS`, run their explicit
check targets:

```sh
ninja -C build check-mock-rest check-trace-qpu
```

For plugin-specific behavior, install steps, and runnable examples, see the
README in each plugin directory.

## Create your own

Copy either directory, rename the YAML/sources/targets, and build with:

```sh
cmake -B build \
  -DCUDAQ_EXTERNAL_PROJECTS="my-plugin" \
  -DCUDAQ_EXTERNAL_MY_PLUGIN_SOURCE_DIR=/path/to/my-plugin
```
