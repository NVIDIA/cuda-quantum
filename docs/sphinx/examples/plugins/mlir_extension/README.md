# Minimal out-of-tree CUDA-Q MLIR Python extension

Minimal out-of-tree example that builds a CUDA-Q MLIR Python extension against
an installed `cudaq-devel` wheel. The script `scripts/validate_devel_wheel.sh`
validates the `cudaq-devel` wheel against this example.

## Layout

The file layout follows standard MLIR conventions:

```txt
include/Trivial/  - public headers for the dialect + pass
lib/              - the pure MLIR library
python/           - the nanobind extension built with the CUDA-Q helpers
cudaq_mlir_extension/  - pip package stub and entry-point declaration
```

The dialect and the `trivial-pass` are defined via TableGen, using the
`mlir-tblgen` binary shipped in the `cudaq-devel` wheel.

## Downstream dialect registration

CUDA-Q discovers out-of-tree MLIR dialects through the `cudaq.mlir_dialects`
entry point group. This example declares one entry point that forwards to
`register_dialects` on the package:

```toml
[project.entry-points."cudaq.mlir_dialects"]
trivial = "cudaq_mlir_extension:register_dialects"
```

After installation, importing `cudaq` seeds every `cudaq.mlir.ir.Context` with
the `trivial` dialect automatically—no explicit registration call is required.

## Building

Install `cudaq-devel` and the matching `cudaq` runtime wheel into your virtual
environment, then install this example as a normal pip package:

```bash
pip install cudaq-devel cudaq
pip install /path/to/examples/plugins/mlir_extension

python -c "import cudaq; from cudaq.mlir.ir import Context; \
  ctx = Context(); assert ctx.dialects['trivial'] is not None"
```

The pip package uses `scikit-build-core` to drive the existing CMake project and
registers the `cudaq.mlir_dialects` entry point declared in `pyproject.toml`.
For development, you can also use `cmake` directly:

```bash
source /path/to/venv/bin/activate  # detects installed cudaq-devel in the venv
cmake -S examples/plugins/mlir_extension -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja
cmake --build build
```

although `pip` or equivalent Python tooling will be required to register the
`cudaq.mlir_dialects` entry point.
