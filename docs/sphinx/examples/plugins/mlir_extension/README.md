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
```

The dialect and the `trivial-pass` are defined via TableGen, using the
`mlir-tblgen` binary shipped in the `cudaq-devel` wheel.

## Building

Install the `cudaq-devel` wheel into the active Python environment with a
normal dependency-resolving install (`pip install cudaq-devel`, possibly with
`--find-links` for local wheelhouses). This branch keeps the published
fallback metadata (approved in onsen #6 note_79265934): `cudaq-devel`
depends on the full matching `cudaq` frontend closure, whose install pulls
`cudaq-core` (the shared providers and `libcudaqMLIR`) and the matching
`cuda-quantum-cuXX` runtime. Development against the SDK toolchain (headers,
TableGen, CMake configs) does not require running CUDA-Q programs; the
dependency policy details and fallback guidance live in
`preview/logical/docs/source/reference/building-against-cudaq.md`. If you are
using non-released versions of CUDA-Q (nightly or custom builds), make sure
your Python package manager knows where to find the matching `cudaq` closure
(the `cudaq` metapackage sdist, `cuda-quantum-cuXX`, and `cudaq-core` at the
same version).

## The MLIR SDK provider libraries

`cudaq-core` ships the shared MLIR/LLVM runtime providers next to
`libcudaqMLIR` under its private prefix (`cudaq_core/cudaq/core`):
`libLLVM`, `libMLIR`, `libMLIRExecutionEngineShared` and `libMLIR-C`.
`libcudaqMLIR` carries only CUDA-Q's own dialect and transform code; the
upstream implementation resolves from the providers. `cudaq-devel` contributes
the development side of the same prefix: `bin/llvm-config`, `mlir-tblgen`,
`FileCheck`, headers, static component archives and the upstream CMake
packages under `lib/cmake/{llvm,mlir}`. Building an extension against
`cudaq::MLIR` links that provider set; extensions must not statically embed
upstream MLIR/LLVM archives into a binary that also loads these shared
libraries.

Then configure. CMake will look for a valid installation of CUDA-Q within the
installed Python packages. Make sure to run this command in the same virtual Python
environment used to install the `cudaq-devel` wheel, or pass the path to the correct
Python interpreter explicitly using the `-DPython3_EXECUTABLE` flag.

```bash
pip install cudaq-devel
pip install 'nanobind>=2.12.0,<3'

site=$(python -c 'import site; print(site.getsitepackages()[0])')
cmake -S examples/plugins/mlir_extension -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja
cmake --build build
cmake --install build --prefix "$site" --component TrivialMLIRPythonModules

python -c "import sys, glob; \
  d=glob.glob('$site/cudaq_mlir_extension/mlir/_mlir_libs')[0]; sys.path.insert(0, d); \
  import _mlirExtension; assert _mlirExtension.run_trivial_pass()"
```
