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

Install the `cudaq-devel` wheel into the active Python environment. If you
are using non-released versions of CUDA-Q (nightly or custom builds), make sure
your Python package manager knows where to find the `cudaq-devel` wheel and a compatible
core `cudaq` wheel.

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
